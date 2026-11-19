from sklearn.metrics import roc_auc_score
def auc(m, n):
    m, n = np.asarray(m, float), np.asarray(n, float)
    return float(roc_auc_score(np.r_[np.ones(len(m)), np.zeros(len(n))], np.r_[m, n]))


def pct(m, n):
    n = np.sort(n); lo = np.searchsorted(n, m, "left"); hi = np.searchsorted(n, m, "right")
    return (lo + 0.5 * (hi - lo)) / len(n)


import json,csv,sys,numpy as np
from scipy.stats import ks_2samp
R=sys.argv[1]; CK=sys.argv[2]
def J(p): return json.load(open(p))

rows={(x['sub'],x['name']):x for x in csv.DictReader(open(CK))}
def cand(s,ep):
    sub='run17a_v6/run17a' if s<=4 else 'run17b_v6/run17b'
    nm='anchor_p10_lr1e-5_ep%d_s%d'%(ep,s)
    return rows[(sub,nm)], J(f"{R}/{sub}/{nm}_raw.json")
for px,P in [('p10_s42',J(f"{R}/run16a_v6/run16a/proxy_p10_raw.json")),('p25_s42',J(f"{R}/run16a_v6/run16a/proxy_p25_raw.json"))]:
    pm,pn=np.array(P["mia"]["members"]["mink"]),np.array(P["mia"]["nonmembers"]["mink"])
    for rule in ('util90_tie5','argmax'):
        chosen=[]
        for s in range(1,9):
            cs=[]
            for ep in (5,10):
                r,raw=cand(s,ep); m,n=np.array(raw["mia"]["members"]["mink"]),np.array(raw["mia"]["nonmembers"]["mink"])
                fq=ks_2samp(raw["forget"]["truth_ratio"],P["forget"]["truth_ratio"]).pvalue; pq=ks_2samp(pct(m,n),pct(pm,pn)).pvalue
                cs.append((ep,min(fq,pq),r))
            if rule=='util90_tie5':
                ok=[c for c in cs if float(c[2]['util'])>=0.9] or cs
                best=max(ok,key=lambda c:(c[1],-c[0]))
            else:
                best=max(cs,key=lambda c:c[1])
            chosen.append((s,best[0],best[2]))
        fq=[float(c[2]['fq']) for c in chosen]; pl=[float(c[2]['pl']) for c in chosen]; ut=[float(c[2]['util']) for c in chosen]; pq=[float(c[2]['pq']) for c in chosen]; li=[float(c[2]['lira']) for c in chosen]
        print(px,rule,'epochs',[c[1] for c in chosen],'FQ',sum(f>.05 for f in fq),'Priv',sum(abs(p)<=.2 for p in pl),'Box',sum(f>.05 and abs(p)<=.2 for f,p in zip(fq,pl)),'RQ',sum(min(f,q)>.05 for f,q in zip(fq,pq)),
              'PL %.2f..%.2f util %.2f..%.2f lira %.3f..%.3f'%(min(pl),max(pl),min(ut),max(ut),min(li),max(li)))
