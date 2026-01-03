import argparse
import random
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    ap.add_argument("--num", type=int, default=50, help="Number of images to visualize")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--thickness", type=int, default=2)
    ap.add_argument("--font_scale", type=float, default=0.6)
    ap.add_argument("--max_labels_per_image", type=int, default=200)
    return ap.parse_args()


def load_yaml(p: Path):
    with p.open("r") as f:
        return yaml.safe_load(f)


def resolve_paths(data_yaml_path: Path, split: str):
    cfg = load_yaml(data_yaml_path)

    base = Path(cfg.get("path", data_yaml_path.parent)).expanduser()
    img_rel = cfg.get(split, None)
    if img_rel is None:
        raise ValueError(f"`{split}` not found in data.yaml")

    img_dir = (base / img_rel).resolve()
    lbl_dir = (base / "labels" / split).resolve()

    # names can be list or dict
    names = cfg.get("names", None)
    if names is None:
        # fallback
        names = [str(i) for i in range(int(cfg.get("nc", 0)))]
    else:
        if isinstance(names, dict):
            # dict keys may be int or str
            # convert to list in index order
            max_k = max(int(k) for k in names.keys())
            lst = [""] * (max_k + 1)
            for k, v in names.items():
                lst[int(k)] = str(v)
            names = lst
        elif isinstance(names, list):
            names = [str(x) for x in names]
        else:
            names = [str(names)]

    return img_dir, lbl_dir, names


def find_images(img_dir: Path):
    imgs = []
    for p in img_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            imgs.append(p)
    return sorted(imgs)


def read_yolo_labels(label_path: Path):
    if not label_path.exists():
        return []
    lines = label_path.read_text().strip().splitlines()
    anns = []
    for ln in lines:
        s = ln.strip().split()
        if len(s) != 5:
            continue
        cls = int(float(s[0]))
        cx, cy, w, h = map(float, s[1:])
        anns.append((cls, cx, cy, w, h))
    return anns


def yolo_to_xyxy(cx, cy, w, h, W, H):
    # normalized -> pixel xyxy
    x1 = (cx - w / 2.0) * W
    y1 = (cy - h / 2.0) * H
    x2 = (cx + w / 2.0) * W
    y2 = (cy + h / 2.0) * H
    x1 = max(0, min(W - 1, int(round(x1))))
    y1 = max(0, min(H - 1, int(round(y1))))
    x2 = max(0, min(W - 1, int(round(x2))))
    y2 = max(0, min(H - 1, int(round(y2))))
    return x1, y1, x2, y2


def draw_annotations(img, anns, names, thickness=2, font_scale=0.6, max_labels=200):
    H, W = img.shape[:2]
    shown = 0
    for (cls, cx, cy, w, h) in anns:
        if shown >= max_labels:
            break
        x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, w, h, W, H)
        # draw bbox (default green)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness)
        # label text
        cname = names[cls] if 0 <= cls < len(names) else str(cls)
        text = f"{cls}:{cname}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        ty = max(0, y1 - 5)
        cv2.rectangle(img, (x1, max(0, ty - th - 4)), (x1 + tw + 4, ty + 2), (0, 255, 0), -1)
        cv2.putText(img, text, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
        shown += 1
    return img


def main():
    args = parse_args()
    random.seed(args.seed)

    data_yaml = Path(args.data).expanduser().resolve()
    img_dir, lbl_dir, names = resolve_paths(data_yaml, args.split)

    if not img_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {img_dir}")
    if not lbl_dir.exists():
        raise FileNotFoundError(f"Label dir not found: {lbl_dir}")

    imgs = find_images(img_dir)
    if len(imgs) == 0:
        raise RuntimeError(f"No images found in: {img_dir}")

    k = min(args.num, len(imgs))
    sampled = random.sample(imgs, k)

    # output dir: same dir as this script file
    script_dir = Path(__file__).resolve().parent
    out_dir = script_dir / "vis_val_jpg"
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(sampled, desc=f"Visualizing {args.split}"):
        im = cv2.imread(str(img_path))
        if im is None:
            continue
        label_path = lbl_dir / f"{img_path.stem}.txt"
        anns = read_yolo_labels(label_path)

        vis = im.copy()
        vis = draw_annotations(
            vis, anns, names,
            thickness=args.thickness,
            font_scale=args.font_scale,
            max_labels=args.max_labels_per_image
        )

        out_path = out_dir / f"{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), vis)

    print(f"Saved {k} visualizations to: {out_dir}")


if __name__ == "__main__":
    main()