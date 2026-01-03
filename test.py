import argparse
import random
from pathlib import Path
import yaml
from ultralytics import YOLO


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True, help="best.pt path")
    ap.add_argument("--data", type=str, required=True, help="data.yaml path")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.60)
    ap.add_argument("--split", type=str, default="test", choices=["val", "test"])
    ap.add_argument("--sample_n", type=int, default=50, help="number of test images for visualization demo")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_project", type=str, default="runs/demo_test", help="output project dir")
    ap.add_argument("--out_name", type=str, default="exp", help="output experiment name")
    return ap.parse_args()


def load_data_yaml(p: Path):
    cfg = yaml.safe_load(p.read_text())
    base = Path(cfg.get("path", p.parent)).expanduser()
    split_rel = cfg.get("test" if cfg.get("test") else "val")  # fallback
    return cfg, base


def get_split_image_dir(cfg, base: Path, split: str):
    key = split
    if key not in cfg:
        raise ValueError(f"'{split}' not found in data.yaml")
    img_dir = (base / cfg[split]).resolve()
    return img_dir


def list_images(img_dir: Path):
    imgs = [p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    return sorted(imgs)


def main():
    args = parse_args()
    random.seed(args.seed)

    weights = Path(args.weights).expanduser().resolve()
    data_yaml = Path(args.data).expanduser().resolve()
    assert weights.exists(), f"weights not found: {weights}"
    assert data_yaml.exists(), f"data.yaml not found: {data_yaml}"

    model = YOLO(str(weights))

    # 1) Evaluate on split (test)
    print(f"\n[1/2] Evaluating on split='{args.split}' ...")
    metrics = model.val(
        data=str(data_yaml),
        split=args.split,
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        plots=True,
        save_json=True,
        project=args.out_project,
        name=f"{args.out_name}_val_{args.split}",
    )
    # metrics already prints a table; this keeps a reference if you want to inspect in python.

    # 2) Prediction demo on sampled images
    cfg, base = load_data_yaml(data_yaml)
    img_dir = get_split_image_dir(cfg, base, args.split)
    if not img_dir.exists():
        raise FileNotFoundError(f"image dir for split '{args.split}' not found: {img_dir}")

    imgs = list_images(img_dir)
    if len(imgs) == 0:
        raise RuntimeError(f"No images found in: {img_dir}")

    k = min(args.sample_n, len(imgs))
    sample_imgs = random.sample(imgs, k)
    # Ultralytics predict 支持传 list
    sample_sources = [str(p) for p in sample_imgs]

    print(f"\n[2/2] Running predict demo on {k} images from {img_dir} ...")
    model.predict(
        source=sample_sources,
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        save=True,
        save_txt=True,
        save_conf=True,
        project=args.out_project,
        name=f"{args.out_name}_pred_{args.split}",
    )

    print("\nDone.")
    print(f"- Val outputs: {Path(args.out_project) / (args.out_name + '_val_' + args.split)}")
    print(f"- Pred outputs: {Path(args.out_project) / (args.out_name + '_pred_' + args.split)}")


if __name__ == "__main__":
    main()


'''
python -m src.test --weights /root/autodl-tmp/code/runs/detect/stage1_m_640/weights/best.pt --data /root/autodl-tmp/data/processed/data.yaml --split val --imgsz 640 --conf 0.25 --iou 0.6 --sample_n 50 --out_project /root/autodl-tmp/code/runs/demo_test --out_name stage1m640

'''
