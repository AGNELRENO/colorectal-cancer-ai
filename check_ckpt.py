import sys, torch
sys.path.insert(0, "src")
from model_rmscnn import RMSCNN

ck = torch.load("checkpoints/best_rmscnn_crc.pth", map_location="cpu", weights_only=True)
sd = ck.get("model_state_dict", ck)

ckpt_params = sum(v.numel() for k, v in sd.items()
                  if v.is_floating_point() and not k.endswith(("running_mean", "running_var")))
model_params = sum(p.numel() for p in RMSCNN().parameters())

print("Checkpoint parameters :", f"{ckpt_params:,}")
print("Your model_rmscnn.py  :", f"{model_params:,}")
print("First keys:", list(sd.keys())[:6])
