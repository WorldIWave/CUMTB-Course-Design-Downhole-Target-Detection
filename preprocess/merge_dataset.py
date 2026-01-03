import argparse, os, hashlib
from pathlib import Path
from tqdm import tqdm
import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def md5_file(p: Path, chunk=1024*1024):
    h = hashlib.md5()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def read_yolo_label(path: Path):
    if not path.exists():
        return []
    lines = path.read_text().strip().splitlines()
    out = []
    for ln in lines:
        s = ln.strip().split()
        if len(s) != 5:
            continue
        cls = int(float(s[0]))
        cx, cy, w, h = map(float, s[1:])
        out.append((cls, cx, cy, w, h))
    return out

def write_yolo_label(path: Path, anns):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for (cls, cx, cy, w, h) in anns:
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""))

def safe_link_or_copy(src: Path, dst: Path, link: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if link:
        os.link(src, dst)
    else:
        import shutil
        shutil.copy2(src, dst)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="data2023_yolo root")
    ap.add_argument("--out", type=str, default="data/processed")
    ap.add_argument("--link", action="store_true", help="use hardlink instead of copy")
    ap.add_argument("--merge_by_hash", action="store_true",
                    help="merge same images across subdatasets by md5 and merge labels")
    ap.add_argument("--make_test_from_val", type=float, default=0.0,
                    help="e.g. 0.5 means split val into val/test half-half by filename (simple)")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

def main():
    args = parse_args()
    root = Path(args.root)
    out = Path(args.out)

    # mapping: folder -> (global_class_id, class_name)
    # 你可按实际文件夹名微调
    mapping = [
        ("coal_miner_data2023_yolo", 0, "coal_miner"),
        ("hydraulic_support_guard_plate _data2023_yolo", 1, "hydraulic_support_guard_plate"),
        ("large_coal_data2023_yolo", 2, "large_coal"),
        ("towline _data2023_yolo", 3, "towline"),
        ("miner_behavior _data2023_yolo", 4, "miner_behavior"),
        ("mine_safety_helmet__data2023_yolo", 5, "mine_safety_helmet"),
    ]

    # output dirs
    for sp in ["train", "val", "test"]:
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)

    # for hash merging
    hash2name = {}   # md5 -> canonical filename (without ext)
    canon_labels = { "train": {}, "val": {}, "test": {} }  # split -> stem -> list of anns

    def handle_one(split, img_path, lbl_path, class_id, prefix):
        if img_path.suffix.lower() not in IMG_EXTS:
            return

        # decide canonical name
        if args.merge_by_hash:
            h = md5_file(img_path)
            if h not in hash2name:
                # new canonical name: keep original stem but add prefix to reduce collision risk
                hash2name[h] = f"{prefix}__{img_path.stem}"
            stem = hash2name[h]
        else:
            # always unique by prefix
            stem = f"{prefix}__{img_path.stem}"

        # copy/link image once
        out_img = out / "images" / split / f"{stem}{img_path.suffix.lower()}"
        safe_link_or_copy(img_path, out_img, args.link)

        # read label and force class id
        anns = read_yolo_label(lbl_path)
        new_anns = []
        for (_, cx, cy, w, h) in anns:
            new_anns.append((class_id, cx, cy, w, h))

        # merge labels if needed
        if stem not in canon_labels[split]:
            canon_labels[split][stem] = []
        canon_labels[split][stem].extend(new_anns)

    # iterate all subdatasets
    for folder, cid, cname in mapping:
        sub = root / folder
        if not sub.exists():
            print(f"[WARN] Not found: {sub} (skip)")
            continue

        for split in ["train", "val"]:
            img_dir = sub / "image" / split
            lbl_dir = sub / "labels" / split
            if not img_dir.exists():
                # 有的可能叫 images 而不是 image
                alt = sub / "images" / split
                if alt.exists():
                    img_dir = alt
            if not img_dir.exists():
                print(f"[WARN] No image dir: {img_dir} (skip)")
                continue

            imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
            prefix = cname
            for img_path in tqdm(imgs, desc=f"{folder}:{split}"):
                lbl_path = (lbl_dir / (img_path.stem + ".txt"))
                handle_one(split, img_path, lbl_path, cid, prefix)

    # optionally split val into val/test
    if args.make_test_from_val and args.make_test_from_val > 0:
        import random
        random.seed(args.seed)
        stems = list(canon_labels["val"].keys())
        random.shuffle(stems)
        n_test = int(len(stems) * float(args.make_test_from_val))
        test_stems = set(stems[:n_test])

        # move selected val items into test (both images and labels)
        for stem in test_stems:
            canon_labels["test"][stem] = canon_labels["val"].pop(stem)

            # move image file
            # find image in val by trying common extensions
            for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                src_img = out / "images" / "val" / f"{stem}{ext}"
                if src_img.exists():
                    dst_img = out / "images" / "test" / src_img.name
                    src_img.replace(dst_img)
                    break

    # write merged labels
    for split in ["train", "val", "test"]:
        for stem, anns in tqdm(canon_labels[split].items(), desc=f"Write labels:{split}"):
            out_lbl = out / "labels" / split / f"{stem}.txt"
            write_yolo_label(out_lbl, anns)

    # write data.yaml
    names = {cid: cname for (_, cid, cname) in mapping}
    data_yaml = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True))
    print("Wrote:", out / "data.yaml")
    print("Done. Output:", out)

if __name__ == "__main__":
    main()

'''
 python -m src.preprocess.merge_dataset --root ../data/raw/DsLMF/data2023_yolo --out ../data/processed --link --merge_by_hash --make_test_from_val 0.2

'''
