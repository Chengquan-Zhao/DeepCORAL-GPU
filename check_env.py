"""检查 GPU 环境，并验证 GPU 数据管线与 torchvision 标准预处理的结果一致。

    python check_env.py
"""
import os

import torch
from PIL import Image
from torchvision import datasets, transforms

import data_gpu

HERE = os.path.dirname(os.path.abspath(__file__))

print(f"torch {torch.__version__}, CUDA 可用: {torch.cuda.is_available()}, 编译时 CUDA 版本: {torch.version.cuda}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name}, {p.total_memory / 2**30:.1f} GB")
    device = torch.device("cuda")
else:
    print("  没有可用的 CUDA。若机器有 NVIDIA 显卡，多半是装了 CPU 版 PyTorch，见 README 重新安装。")
    device = torch.device("cpu")

# 用 dslr 的前 8 张图，对比 GPU 管线与 torchvision transforms 的输出
root = os.path.join(HERE, "data", "office31")
ds = datasets.ImageFolder(os.path.join(root, "dslr", "images"))
paths = [p for p, _ in ds.samples[:8]]
dom = data_gpu.GPUDomain({"images": torch.stack([torch.from_numpy(data_gpu._load(p)) for p in paths]),
                          "labels": torch.tensor(ds.targets[:8]), "classes": ds.classes}, device)
ref_tf = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.ToTensor(),
                             transforms.Normalize(data_gpu._MEAN, data_gpu._STD)])
ref = torch.stack([ref_tf(Image.open(p).convert("RGB")) for p in paths])
ours, _ = next(dom.test_batches(8))
print(f"中心裁剪 与 torchvision 最大误差: {(ours.cpu() - ref).abs().max().item():.2e}")

# 随机裁剪 + 翻转：逐张用切片重算一遍，应完全一致
g = torch.Generator(device=device).manual_seed(0)
idx = torch.arange(8, device=device)
aug, _ = dom._augment(idx, g)
g = torch.Generator(device=device).manual_seed(0)
oy = torch.randint(0, 33, (8, 1), device=device, generator=g)
ox = torch.randint(0, 33, (8, 1), device=device, generator=g)
flip = torch.rand(8, 1, device=device, generator=g) < 0.5
for i in range(8):
    crop = dom.images[i, oy[i]:oy[i] + 224, ox[i]:ox[i] + 224]
    crop = crop.flip(1) if flip[i] else crop
    assert torch.equal(aug[i], dom._normalize(crop[None])[0])
print(f"随机裁剪/翻转 校验通过（本次 8 张中翻转 {flip.sum().item()} 张）")
