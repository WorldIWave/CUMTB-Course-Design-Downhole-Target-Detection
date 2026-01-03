import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from ultralytics import YOLO
from torch.utils.tensorboard import SummaryWriter



def parse_args():
    ap = argparse.ArgumentParser()

    # core
    ap.add_argument("--data", type=str, default="data/processed/data.yaml")
    ap.add_argument("--model", type=str, default="yolov8m.pt")
    ap.add_argument("--imgsz", type=int, default=856)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--project", type=str, default="runs/detect")
    ap.add_argument("--name", type=str, default="longwall_yolov8")

    # training knobs
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--close_mosaic", type=int, default=10)
    ap.add_argument("--amp", action="store_true", default=True)

    # precision targeting
    ap.add_argument("--precision_target", type=float, default=0.98,
                    help="Target precision threshold for acceptance (operational).")
    ap.add_argument("--conf_oper", type=float, default=0.7,
                    help="Operational confidence threshold used to compute precision/recall during training.")
    ap.add_argument("--iou_oper", type=float, default=0.60,
                    help="IoU threshold used in operational validation.")
    ap.add_argument("--conf_sweep_every", type=int, default=5,
                    help="If >0, run conf sweep every N epochs to find best conf reaching precision_target.")
    ap.add_argument("--conf_sweep_min", type=float, default=0.3)
    ap.add_argument("--conf_sweep_max", type=float, default=0.95)
    ap.add_argument("--conf_sweep_steps", type=int, default=25)

    return ap.parse_args()


def _extract_pr(metrics):
    """
    Ultralytics metrics object -> (precision, recall, map50, map5095)
    Keys can vary slightly by version; we robustly try common ones.
    """
    rd = getattr(metrics, "results_dict", {}) or {}
    def g(*keys):
        for k in keys:
            if k in rd:
                try:
                    return float(rd[k])
                except Exception:
                    pass
        return float("nan")

    p = g("metrics/precision(B)", "metrics/precision", "precision")
    r = g("metrics/recall(B)", "metrics/recall", "recall")
    map50 = g("metrics/mAP50(B)", "metrics/mAP50", "map50")
    map5095 = g("metrics/mAP50-95(B)", "metrics/mAP50-95", "map")
    return p, r, map50, map5095


def main():
    args = parse_args()

    model = YOLO(args.model)

    # --------- custom logging holders ----------
    state = {
        "writer": None,
        "run_dir": None,
        "history": [],  # list of dict rows (epoch-wise)
    }
    def _get_metric(d, *keys):
        for k in keys:
            if k in d:
                try:
                    return float(d[k])
                except Exception:
                    pass
        return float("nan")

    def on_train_start(trainer):
        # trainer.save_dir like runs/detect/exp_name
        run_dir = Path(trainer.save_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        state["run_dir"] = run_dir
        state["writer"] = SummaryWriter(log_dir=str(run_dir))

        print(f"[CB] TensorBoard logdir = {run_dir}")

    def on_fit_epoch_end(trainer):
        epoch = int(getattr(trainer, "epoch", -1)) + 1
        run_dir = state["run_dir"]
        writer = state["writer"]

        # 从 results.csv 读取
        csv_path = run_dir / "results.csv"
        if not csv_path.exists():
            return

        df = pd.read_csv(csv_path)
        if len(df) == 0:
            return
        row = df.iloc[-1].to_dict()

        def pick(*keys):
            for k in keys:
                if k in row and pd.notna(row[k]):
                    return float(row[k])
            return float("nan")

        p = pick("metrics/precision(B)", "metrics/precision")
        r = pick("metrics/recall(B)", "metrics/recall")
        map50 = pick("metrics/mAP50(B)", "metrics/mAP50")
        map5095 = pick("metrics/mAP50-95(B)", "metrics/mAP50-95")

        if writer:
            writer.add_scalar("val/precision", p, epoch)
            writer.add_scalar("val/recall", r, epoch)
            writer.add_scalar("val/mAP50", map50, epoch)
            writer.add_scalar("val/mAP50-95", map5095, epoch)

    def on_train_end(trainer):
        if state["writer"]:
            state["writer"].flush()
            state["writer"].close()
    

    # --------- register callbacks ----------
    # Ultralytics supports model.add_callback(event, func)
    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    model.add_callback("on_train_end", on_train_end)

    # --------- train ----------
    model.train(
        cache="disk",
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,

        val=True,
        plots=True,

        patience=args.patience,
        amp=args.amp,
        close_mosaic=args.close_mosaic,

        # common aug knobs (you可以按需要再调)
        mosaic=1.0,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=2.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
    )


if __name__ == "__main__":
    main()


'''
python -m src.train --data /root/autodl-tmp/data/processed/data.yaml --model yolov8m.pt --imgsz 640 --epochs 30 --batch 16 --workers 16 --device 0 --name stage1_m_640

tensorboard --logdir /root/autodl-tmp/code/runs/detect/stage1_m_640/ --port 6006 --host 127.0.0.1
'''
