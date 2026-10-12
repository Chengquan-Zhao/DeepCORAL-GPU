"""GPU 数据管线：图片只解码一次并缓存，训练时整个数据集常驻显存，裁剪/翻转/归一化都在 GPU 上做。

预处理与 CPU 版完全相同：缩放到 256x256 → 训练时随机裁剪 224 + 水平翻转 / 测试时中心裁剪 → ImageNet 归一化。
第一次运行时把解码、缩放后的 uint8 图片缓存到 data/cache/，之后的运行直接读缓存（几秒）。
显存占用：amazon ≈ 554MB，webcam ≈ 156MB，dslr ≈ 98MB（只加载当前用到的两个域）。
"""
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image
from torchvision import datasets

DOMAINS = {"A": "amazon", "D": "dslr", "W": "webcam"}
RESIZE, CROP = 256, 224
_MEAN, _STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def _load(path):
    with Image.open(path) as im:   # 与 transforms.Resize((256,256)) 对 PIL 图片的处理一致（双线性 + 抗锯齿）
        return np.array(im.convert("RGB").resize((RESIZE, RESIZE), Image.BILINEAR))


def load_domain(root, domain, cache_dir):
    """返回 {"images": uint8 [N,256,256,3], "labels": int64 [N], "classes": list}，带磁盘缓存。"""
    name = DOMAINS.get(domain.upper(), domain)
    cache = os.path.join(cache_dir, f"{name}_{RESIZE}.pt")
    if os.path.exists(cache):
        return torch.load(cache)
    folder = os.path.join(root, name, "images")
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"找不到 {folder}，请先运行 python download_office31.py")
    ds = datasets.ImageFolder(folder)
    print(f"首次运行：解码并缓存 {name}（{len(ds)} 张）...", flush=True)
    with ThreadPoolExecutor(min(16, os.cpu_count() or 4)) as ex:
        imgs = list(ex.map(_load, [p for p, _ in ds.samples]))
    data = {"images": torch.from_numpy(np.stack(imgs)), "labels": torch.tensor(ds.targets),
            "classes": ds.classes}
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(data, cache)
    return data


class GPUDomain:
    def __init__(self, data, device):
        self.images = data["images"].to(device)          # NHWC uint8，常驻显存
        self.labels = data["labels"].to(device)
        self.classes = data["classes"]
        self.device = device
        self.mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1) * 255
        self.std = torch.tensor(_STD, device=device).view(1, 3, 1, 1) * 255
        self._ar = torch.arange(CROP, device=device)

    def __len__(self):
        return self.labels.numel()

    def _normalize(self, x):
        # NHWC 连续内存 permute 成 NCHW 后，正好是 channels_last 布局，卷积在 GPU 上更快
        return (x.permute(0, 3, 1, 2).float() - self.mean) / self.std

    def _augment(self, idx, g):
        """每张图独立的随机裁剪 + 随机水平翻转，用一次 gather 完成。"""
        b = idx.numel()
        oy = torch.randint(0, RESIZE - CROP + 1, (b, 1), device=self.device, generator=g)
        ox = torch.randint(0, RESIZE - CROP + 1, (b, 1), device=self.device, generator=g)
        flip = torch.rand(b, 1, device=self.device, generator=g) < 0.5
        iy = oy + self._ar
        ix = torch.where(flip, ox + (CROP - 1 - self._ar), ox + self._ar)
        x = self.images[idx[:, None, None], iy[:, :, None], ix[:, None, :]]   # [B,224,224,3]
        return self._normalize(x), self.labels[idx]

    def train_batches(self, batch_size, g):
        """无限迭代：每个 epoch 重新打乱，丢弃最后不足一个 batch 的部分（同 DataLoader drop_last）。"""
        n = len(self)
        while True:
            perm = torch.randperm(n, device=self.device, generator=g)
            for i in range(0, n - batch_size + 1, batch_size):
                yield self._augment(perm[i:i + batch_size], g)

    def test_batches(self, batch_size):
        o = (RESIZE - CROP) // 2
        for i in range(0, len(self), batch_size):
            yield (self._normalize(self.images[i:i + batch_size, o:o + CROP, o:o + CROP]),
                   self.labels[i:i + batch_size])
