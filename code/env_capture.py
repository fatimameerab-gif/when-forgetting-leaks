import platform
import subprocess
import sys

print("python", sys.version.replace("\n", " "))
print("platform", platform.platform())
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.version.cuda)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print("gpu", i, torch.cuda.get_device_name(i))
except ImportError:
    print("torch not installed")
for mod in ("transformers", "datasets", "accelerate", "bitsandbytes", "rouge_score", "scipy",
            "sklearn", "numpy", "pandas", "matplotlib"):
    try:
        m = __import__(mod)
        print(mod, getattr(m, "__version__", "?"))
    except ImportError:
        print(mod, "not installed")
print("---- pip freeze ----")
print(subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout)
