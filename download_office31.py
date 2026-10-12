"""下载并解压 Office-31 数据集到 data/office31/。

来源：Saenko et al. (ECCV 2010) 官方发布的 domain_adaptation_images.tar.gz（Google Drive，约 77MB）。
另外顺带缓存 torchvision 的 ImageNet 预训练 AlexNet 权重。
如果 Google Drive 无法访问，可手动下载该压缩包放到 data/ 下再运行本脚本。
"""
import os
import tarfile

GDRIVE_ID = "0B4IapRTv9pJ1WGZVd1VDMmhwdlE"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
ARCHIVE = os.path.join(DATA_DIR, "domain_adaptation_images.tar.gz")
OUT_DIR = os.path.join(DATA_DIR, "office31")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    if all(os.path.isdir(os.path.join(OUT_DIR, d, "images")) for d in ("amazon", "dslr", "webcam")):
        print("Office-31 已存在：", OUT_DIR)
    else:
        if not os.path.exists(ARCHIVE):
            import gdown
            gdown.download(id=GDRIVE_ID, output=ARCHIVE, quiet=False)
        print("解压中 ...")
        with tarfile.open(ARCHIVE) as tar:
            tar.extractall(OUT_DIR, filter="data")
    for d in ("amazon", "dslr", "webcam"):
        root = os.path.join(OUT_DIR, d, "images")
        n = sum(len(files) for _, _, files in os.walk(root))
        print(f"{d:7s}: {len(os.listdir(root))} 类, {n} 张")   # 预期 2817 / 498 / 795 张

    from torchvision.models import AlexNet_Weights, alexnet
    alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
    print("AlexNet 预训练权重已缓存")


if __name__ == "__main__":
    main()
