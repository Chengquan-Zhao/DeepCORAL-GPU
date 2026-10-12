"""Deep CORAL 训练脚本 —— GPU 加速版（复现 Sun & Saenko, ECCV-W 2016）。

与 CPU 版的算法、超参数完全相同，区别只在实现效率：
  1. 数据集常驻显存，增广在 GPU 上做（data_gpu.py），没有 DataLoader 瓶颈；
  2. 源域、目标域拼成一个 batch 做一次前向（AlexNet 没有 BN，结果与两次前向等价）；
  3. 混合精度 AMP（fp16）+ channels_last + cudnn.benchmark；CORAL 损失仍用 fp32 计算；
  4. 训练循环里不调用 .item()，只在记录日志时同步一次 GPU。

例：
    python train.py --source A --target W                    # Deep CORAL
    python train.py --source A --target W --lambda_coral 0   # 基线 CNN（无适配）
"""
import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from coral import coral_loss
from data_gpu import GPUDomain, load_domain
from model import DeepCORAL

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description="Deep CORAL on Office-31 (GPU)")
    p.add_argument("--data_root", default=os.path.join(HERE, "data", "office31"))
    p.add_argument("--cache_dir", default=os.path.join(HERE, "data", "cache"))
    p.add_argument("--source", default="A", help="A / D / W")
    p.add_argument("--target", default="W", help="A / D / W")
    # 以下默认值均取自论文第 4 节
    p.add_argument("--batch_size", type=int, default=128, help="源域、目标域各取一个 batch")
    p.add_argument("--lr", type=float, default=1e-3, help="base learning rate")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--lambda_coral", type=float, default=1.0,
                   help="CORAL 损失权重；论文的原则是让训练结束时分类损失≈CORAL 损失，设 0 即无适配基线")
    p.add_argument("--iters", type=int, default=2000, help="论文 Fig.2 训练了约 2000 次迭代")
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                   help="混合精度训练（默认开启，--no-amp 关闭以得到纯 fp32 结果）")
    p.add_argument("--gpu", type=int, default=0, help="使用第几块 GPU")
    p.add_argument("--out_dir", default=os.path.join(HERE, "results"))
    return p.parse_args()


def param_groups(model, base_lr, wd):
    """沿用 Caffe 的逐层设置：权重 lr×1、decay×1；偏置 lr×2、decay×0；fc8 再整体 ×10（从头训练）。"""
    groups = []
    for name, p in model.named_parameters():
        mult = 10.0 if name.startswith("fc8") else 1.0
        if name.endswith("bias"):
            groups.append({"params": [p], "lr": base_lr * mult * 2, "weight_decay": 0.0})
        else:
            groups.append({"params": [p], "lr": base_lr * mult, "weight_decay": wd})
    return groups


@torch.no_grad()
def evaluate(model, domain, amp):
    model.eval()
    correct = torch.zeros((), device=domain.device)
    for x, y in domain.test_batches(512):
        with torch.autocast(domain.device.type, dtype=torch.float16, enabled=amp):
            correct += (model(x).argmax(1) == y).sum()
    model.train()
    return correct.item() / len(domain)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda", args.gpu)
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
        dev_name = torch.cuda.get_device_name(device)
    else:
        device, args.amp, dev_name = torch.device("cpu"), False, "CPU"
        print("警告：没有检测到 CUDA，将在 CPU 上运行（很慢，仅用于调试）。"
              "请安装 CUDA 版 PyTorch，见 README。")
    shift = f"{args.source.upper()}2{args.target.upper()}"
    tag = f"{shift}_lambda{args.lambda_coral:g}_seed{args.seed}"
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[{tag}] device={dev_name}, amp={args.amp}")

    src = GPUDomain(load_domain(args.data_root, args.source, args.cache_dir), device)
    # 无监督协议：目标域全部图片（不带标签）参与训练，同样的全部图片用于测试
    tgt = GPUDomain(load_domain(args.data_root, args.target, args.cache_dir), device)
    assert src.classes == tgt.classes
    print(f"source={len(src)} imgs, target={len(tgt)} imgs, classes={len(src.classes)}")

    model = DeepCORAL(len(src.classes)).to(device, memory_format=torch.channels_last)
    optimizer = torch.optim.SGD(param_groups(model, args.lr, args.weight_decay), lr=args.lr,
                                momentum=args.momentum)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp)

    log_path = os.path.join(args.out_dir, tag + ".csv")
    fields = ["iter", "class_loss", "coral_loss", "train_acc", "test_acc", "elapsed_s"]
    log_f = open(log_path, "w", newline="")
    writer = csv.DictWriter(log_f, fieldnames=fields)
    writer.writeheader()

    def log_row(it, cls, cor, tr_acc, t0):
        acc = evaluate(model, tgt, args.amp)
        row = {"iter": it, "class_loss": cls, "coral_loss": cor, "train_acc": tr_acc,
               "test_acc": acc, "elapsed_s": round(time.time() - t0, 1)}
        writer.writerow(row)
        log_f.flush()
        print(f"[{tag}] iter {it:5d} | class {cls:.4f} | coral {cor:.4f} | "
              f"src acc {tr_acc:.4f} | tgt acc {acc:.4f} | {row['elapsed_s']}s", flush=True)
        return acc

    t0 = time.time()
    log_row(0, float("nan"), float("nan"), float("nan"), t0)
    g = torch.Generator(device=device).manual_seed(args.seed)
    src_iter, tgt_iter = src.train_batches(args.batch_size, g), tgt.train_batches(args.batch_size, g)
    zero = lambda: torch.zeros((), device=device)
    sum_cls, sum_cor, n_correct, n_seen = zero(), zero(), zero(), 0
    model.train()
    for it in range(1, args.iters + 1):
        xs, ys = next(src_iter)
        xt, _ = next(tgt_iter)                  # 目标域标签不使用
        b = xs.size(0)

        with torch.autocast(device.type, dtype=torch.float16, enabled=args.amp):
            out = model(torch.cat([xs, xt]))    # 两路共享参数，一次前向
        out = out.float()                       # 损失（尤其是协方差）用 fp32 计算
        out_s, out_t = out[:b], out[b:]
        loss_cls = F.cross_entropy(out_s, ys)
        loss_cor = coral_loss(out_s, out_t)     # λ=0 时也计算，用于画 Fig.2(c) 的距离曲线
        loss = loss_cls + args.lambda_coral * loss_cor   # Eq.(6)，t=1（只在 fc8 上加）

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        sum_cls += loss_cls.detach()
        sum_cor += loss_cor.detach()
        n_correct += (out_s.argmax(1) == ys).sum()
        n_seen += b
        if it % args.eval_every == 0 or it == args.iters:
            k = it - (it - 1) // args.eval_every * args.eval_every
            final_acc = log_row(it, sum_cls.item() / k, sum_cor.item() / k,
                                n_correct.item() / n_seen, t0)
            sum_cls, sum_cor, n_correct, n_seen = zero(), zero(), zero(), 0

    log_f.close()
    summary = {"shift": shift, "lambda": args.lambda_coral, "seed": args.seed,
               "iters": args.iters, "amp": args.amp, "target_acc": final_acc}
    with open(os.path.join(args.out_dir, "summary.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")
    print(f"[{tag}] 完成，最终目标域准确率 {final_acc * 100:.1f}%，用时 {time.time() - t0:.0f}s，日志：{log_path}")


if __name__ == "__main__":
    main()
