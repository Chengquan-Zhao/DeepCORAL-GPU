# Deep CORAL 复现（PyTorch · GPU 版）

本仓库是论文 **Deep CORAL: Correlation Alignment for Deep Domain Adaptation**（Baochen Sun & Kate Saenko，
ECCV 2016 Workshops）的 PyTorch 复现，在 Office-31 数据集上做无监督域适应。仓库里**已经包含 Office-31 数据集**，
下载代码后不需要再单独下载数据。

- `coral.py`：CORAL 损失（论文 Eq.1–3），运行 `python coral.py` 会用论文 Eq.4 的解析梯度做自检
- `model.py`：ImageNet 预训练 AlexNet，最后一层 fc8 换成 31 类，权重用 N(0, 0.005) 初始化
- `data_gpu.py`：GPU 数据管线（数据集常驻显存，随机裁剪、翻转、归一化在 GPU 上完成）
- `train.py`：训练和评估（论文 Eq.6）
- `plot.py`：画论文 Fig.2，并把结果和论文 Table 1 列在一起对比
- `run_all.sh` / `run_all.ps1`：一键跑完整的 Table 1
- `check_env.py`：检查 GPU 环境，并校验数据管线
- `data/office31/`：Office-31 数据集，包括 amazon 2817 张、dslr 498 张、webcam 795 张，共 31 类

---

## 运行步骤

下面的命令在 **Linux / macOS 终端** 或 **Windows 的 PowerShell** 中都可以直接用；两者有差别的地方分别列出。

### 第 0 步：准备工作

需要一台装有 **NVIDIA 显卡** 的电脑或服务器，并且已经安装了以下软件：

1. **显卡驱动**：在终端运行下面的命令，能看到显卡型号和 `CUDA Version` 就说明驱动正常。
   ```bash
   nvidia-smi
   ```
2. **Git**：运行下面的命令能看到版本号即可。没有的话，Windows 从 https://git-scm.com/download/win 下载安装；
   Ubuntu 运行 `sudo apt install git`。
   ```bash
   git --version
   ```
3. **Python 3.9 及以上**。推荐用 Anaconda / Miniconda（https://docs.conda.io/en/latest/miniconda.html）管理环境。

### 第 1 步：把代码和数据下载到本地

```bash
git clone https://github.com/Chengquan-Zhao/DeepCORAL-GPU.git
```
```bash
cd DeepCORAL-GPU
```

大约要下载 80MB，主要是数据集。如果 `git clone` 很慢或失败，也可以在仓库网页上点 **Code → Download ZIP**，
下载后解压，再用 `cd` 进入解压出来的文件夹。

### 第 2 步：创建 Python 环境（推荐）

使用 conda：
```bash
conda create -n deepcoral python=3.11 -y
```
```bash
conda activate deepcoral
```

如果不用 conda，也可以用 venv：
```bash
python -m venv .venv
```
然后激活它。Linux / macOS 用第一条命令，Windows PowerShell 用第二条：
```bash
source .venv/bin/activate
```
```bash
.venv\Scripts\Activate.ps1
```

### 第 3 步：安装 CUDA 版 PyTorch 和其它依赖

打开 https://pytorch.org/get-started/locally/ ，依次选择 Stable、你的系统、Pip、Python，以及一个
**不高于 `nvidia-smi` 显示的 CUDA Version** 的 CUDA 版本，然后复制页面给出的命令运行。比如 CUDA 12.8 对应：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```
接着安装其余依赖：
```bash
pip install -r requirements.txt
```

> ⚠️ 不要直接运行 `pip install torch`，在 Windows 上它往往会装成 CPU 版，结果 GPU 用不上。

### 第 4 步：检查环境

```bash
python check_env.py
```
正常的输出应该类似下面这样：
```
torch 2.x.x+cu128, CUDA 可用: True, 编译时 CUDA 版本: 12.8
  GPU 0: NVIDIA GeForce RTX ..., 24.0 GB
中心裁剪 与 torchvision 最大误差: 2.38e-07
随机裁剪/翻转 校验通过（本次 8 张中翻转 3 张）
```
如果显示 `CUDA 可用: False`，说明装的是 CPU 版 PyTorch，需要回到第 3 步重新安装。

### 第 5 步：训练（以 Amazon → Webcam 为例）

```bash
python train.py --source A --target W
```
```bash
python train.py --source A --target W --lambda_coral 0
```
第一条命令训练 Deep CORAL，第二条是不做域适应的基线（论文里的 CNN）。`A` / `D` / `W` 分别代表
amazon / dslr / webcam。

- 第一次运行会先把图片解码、缓存到 `data/cache/`，需要几十秒；之后的运行直接读缓存。
- 每 100 次迭代打印一行日志，包括分类损失、CORAL 损失、源域训练准确率和目标域测试准确率。日志同时保存到
  `results/A2W_lambda1_seed0.csv`，最终准确率追加到 `results/summary.jsonl`。

### 第 6 步：画图，并和论文对比

```bash
python plot.py --shift A2W --lambda_coral 1 --seed 0
```
```bash
python plot.py --table
```
第一条命令复现论文 Fig.2，图片保存为 `results/fig2_A2W.png`；第二条把已经跑完的结果和论文 Table 1 列在一起对比。

### 第 7 步（可选）：复现完整的 Table 1

总共 6 个迁移方向 × 5 个随机种子 × 有/无 CORAL，一共 60 次运行。跑完会自动打印对比表并画图。

Linux：
```bash
bash run_all.sh
```
Windows PowerShell：
```bash
powershell -ExecutionPolicy Bypass -File run_all.ps1
```
在 Linux 上可以用环境变量修改参数，例如 `LAMBDA=0.5 SEEDS="0 1" GPU=1 bash run_all.sh`。

---

## 常用参数（`train.py`）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--source` / `--target` | `A` / `W` | 源域、目标域：`A` `D` `W` |
| `--lambda_coral` | `1.0` | CORAL 损失权重，设为 `0` 就是无适配基线 |
| `--iters` | `2000` | 训练迭代次数 |
| `--batch_size` | `128` | 源域和目标域每次各取多少张 |
| `--lr` | `1e-3` | base learning rate |
| `--seed` | `0` | 随机种子 |
| `--gpu` | `0` | 使用第几块显卡 |
| `--no-amp` | 默认开启 AMP | 关闭混合精度，用纯 fp32 训练 |
| `--eval_every` | `100` | 每多少次迭代评估一次并记录日志 |

## 与论文设置的对应

| 论文 | 本实现 |
|---|---|
| CORAL 损失加在 fc8（31 维）上 | 同 |
| fc8 用 N(0, 0.005) 初始化，学习率是其它层的 10 倍 | 同 |
| 其它层用 ImageNet 预训练参数，沿用原来的逐层设置 | 按 Caffe 惯例：偏置的学习率 ×2，不加 weight decay |
| batch 128，base lr 1e-3，weight decay 5e-4，momentum 0.9 | 同（源域、目标域每次各 128 张） |
| 约 2000 次迭代（Fig.2 的横轴） | `--iters 2000` |
| 用全部带标签的源域数据和全部无标签的目标域数据训练，在目标域上测试 | 同 |
| λ 的选取原则：训练结束时分类损失 ≈ CORAL 损失 | 默认 `1`。可以看 CSV 最后几行的 `class_loss` 和 `coral_loss`，相差很大就调整 λ |
| Caffe + BVLC CaffeNet | PyTorch + torchvision AlexNet（没有 LRN 层，通道数略有不同），输入裁剪为 224 而不是 227 |

GPU 加速相关的实现：数据集常驻显存、源域和目标域拼成一个 batch 一次前向（AlexNet 没有 BN，结果与分两次前向等价）、
AMP 混合精度、channels_last、cudnn.benchmark。CORAL 损失始终用 fp32 计算。

## 论文结果（目标域准确率 %）

| | A→D | A→W | D→A | D→W | W→A | W→D | AVG |
|---|---|---|---|---|---|---|---|
| CNN（无适配） | 63.8 | 61.6 | 51.1 | 95.4 | 49.8 | 99.0 | 70.1 |
| D-CORAL | 66.8 | 66.4 | 52.8 | 95.7 | 51.5 | 99.2 | 72.1 |

由于框架和网络结构有细微差别，复现结果和论文相差一两个百分点属于正常范围。

## 常见问题

- **`CUDA 可用: False`**：装成了 CPU 版 PyTorch。先运行 `pip uninstall torch torchvision -y`，再按第 3 步重新安装。
- **显存不足（CUDA out of memory）**：可以加 `--batch_size 64`，但这会偏离论文设置。估算显存需求大约 3GB。
- **数据丢失或损坏**：运行 `python download_office31.py`，会从官方 Google Drive 重新下载 Office-31。
- **没有显卡**：脚本会自动退回 CPU 运行，可以用来调试，但非常慢（每次迭代需要几秒）。

## 引用与数据来源

```bibtex
@inproceedings{sun2016deep,
  title={Deep CORAL: Correlation Alignment for Deep Domain Adaptation},
  author={Sun, Baochen and Saenko, Kate},
  booktitle={ECCV Workshops},
  year={2016}
}
```
Office-31 数据集来自 Saenko et al., *Adapting Visual Category Models to New Domains*（ECCV 2010），
版权归原作者所有，这里只为方便学术复现而附带，请仅用于研究目的。
