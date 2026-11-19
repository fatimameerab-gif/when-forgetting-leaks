#!/usr/bin/env bash
set -uo pipefail

MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
FORGET=${FORGET:-forget10}
RETAIN=${RETAIN:-retain90}
DATA=${DATA:-data}
FT_EPOCHS=${FT_EPOCHS:-5}
UL_EPOCHS=${UL_EPOCHS:-5}
BS=${BS:-8};       GA=${GA:-4}
UL_BS=${UL_BS:-4}; UL_GA=${UL_GA:-8}
EXTRAS=${EXTRAS:-1}
KEEP_DIR=${KEEP_DIR:-}
PREV_CKPT=${PREV_CKPT:-}
CK=ckpt; R=results
mkdir -p $CK $R logs
[ -z "$KEEP_DIR" ] && [ -d /kaggle/working ] && KEEP_DIR=/kaggle/working/keep_ckpt
if [ -z "$PREV_CKPT" ] && [ -d /kaggle/input ]; then
  hit=$(find /kaggle/input -path "*keep_ckpt/A_original/train_stats.json" 2>/dev/null | head -1)
  [ -n "$hit" ] && PREV_CKPT=$(dirname "$(dirname "$hit")")
fi

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NGPU" -lt 1 ] && NGPU=1
G0=0; G1=$(( NGPU > 1 ? 1 : 0 ))
DD="--data_dir $DATA"
FAILED=()

say ()  { echo "[$(date +%H:%M:%S)] $*"; }
done_ck () { [ -f "$CK/$1/train_stats.json" ]; }
done_ev () { [ -f "$R/$1.json" ]; }
run ()  {
  local g=$1 name=$2; shift 2
  say "START  $name  (GPU $g)"
  if CUDA_VISIBLE_DEVICES=$g "$@" > logs/$name.log 2>&1; then say "DONE   $name"
  else say "FAILED $name  -> see logs/$name.log"; echo "$name" >> logs/_failed.txt; tail -5 logs/$name.log; fi
}
rm -f logs/_failed.txt
say "GPUs: $NGPU   model: $MODEL   forget: $FORGET"

[ -d "$DATA" ] || python download_data.py --out $DATA > logs/download.log 2>&1

if [ -n "$PREV_CKPT" ]; then
  for m in A_original B0_retrain; do
    if ! done_ck $m && [ -f "$PREV_CKPT/$m/train_stats.json" ]; then
      say "reusing $m from $PREV_CKPT"; cp -r "$PREV_CKPT/$m" $CK/
    fi
  done
fi

ft ()  { done_ck $2 || run $1 $2 python train_ft.py --model $MODEL --subset $3 --out $CK/$2 $DD \
           --epochs $FT_EPOCHS --bs $BS --grad_accum $GA; }
unl () { local g=$1 name=$2 method=$3; shift 3
         done_ck A_original || { say "SKIP $name (no Model A)"; return; }
         done_ck $name || run $g $name python unlearn.py --model $CK/A_original --method $method \
           --out $CK/$name --forget $FORGET --retain $RETAIN $DD --epochs $UL_EPOCHS \
           --bs $UL_BS --grad_accum $UL_GA "$@"; }
ev ()  { local g=$1 name=$2; shift 2; local src=${name%_4bit}
         done_ck $src || { say "SKIP eval $name (no checkpoint)"; return; }
         done_ev $name || run $g eval_$name python evaluate.py --model $CK/$src --name $name \
           --forget $FORGET $DD --ref_raw $R/B0_retrain_raw.json "$@"; }

say "=== phase 1: Model A + Oracle (retrain without forget set)"
ft $G0 A_original full &
[ $NGPU -gt 1 ] || wait
ft $G1 B0_retrain $RETAIN &
wait
if [ -n "$KEEP_DIR" ]; then
  mkdir -p "$KEEP_DIR"
  for m in A_original B0_retrain; do
    done_ck $m && [ ! -d "$KEEP_DIR/$m" ] && cp -r $CK/$m "$KEEP_DIR/" && say "kept $m -> $KEEP_DIR"
  done
fi

say "=== phase 2: unlearning"
lane0 () {
  unl $G0 B_ft_retain ft_retain --epochs 1
  unl $G0 C_graddiff  graddiff --lr 1e-5
  [ "$EXTRAS" = 1 ] && unl $G0 E_ga ga --lr 1e-5
  true
}
lane1 () {
  ev  $G1 B0_retrain
  unl $G1 D_npo_rt npo_rt --lr 1e-5 --beta 0.1
  [ "$EXTRAS" = 1 ] && unl $G1 F_simnpo simnpo --lr 1e-5 --beta 2.5
  true
}
if [ $NGPU -gt 1 ]; then lane0 & lane1 & wait; else lane1; lane0; fi

say "=== phase 3: evaluation"
ev $G0 B0_retrain
lane0 () { ev $G0 A_original; ev $G0 B_ft_retain; ev $G0 C_graddiff
           [ "$EXTRAS" = 1 ] && { ev $G0 E_ga; ev $G0 A_original_4bit --load_4bit; }; true; }
lane1 () { ev $G1 D_npo_rt
           [ "$EXTRAS" = 1 ] && { ev $G1 F_simnpo; ev $G1 C_graddiff_4bit --load_4bit
                                  ev $G1 D_npo_rt_4bit --load_4bit; }; true; }
if [ $NGPU -gt 1 ]; then lane0 & lane1 & wait; else lane0; lane1; fi

say "=== phase 4: aggregate"
python aggregate.py --dir $R > logs/aggregate.log 2>&1 && cat $R/summary.md || tail -5 logs/aggregate.log
if [ -f logs/_failed.txt ]; then say "FINISHED WITH FAILURES: $(tr '\n' ' ' < logs/_failed.txt)"
else say "FINISHED - all steps OK"; fi
echo "Zip results/ and logs/ (not ckpt/) and send them back."
