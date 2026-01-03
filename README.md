# 数据预处理与训练说明文档

## 数据预处理步骤

### 1. 下载并解压数据集

- 下载 `MsLMF.7z`
- 解压命令：  
  ```bash
  7z x "MsLMF.7z"
  ```
- 删除 COCO 数据及压缩包：  
  ```bash
  rm MsLMF.7z
  cd MsLMF
  rm -r data2023_coco
  ```

### 2. 配置环境

使用 Python 3.10，安装相关依赖：  
```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
pip install ultralytics opencv-python matplotlib pandas pyyaml tqdm
```

### 3. 合并原始数据集并划分训练集、验证集和测试集

运行 `src/preprocess` 目录下的 `merge_dataset.py`：  
```bash
python -m src.preprocess.merge_dataset --root ../data/raw/DsLMF/data2023_yolo --out ../data/processed --link --merge_by_hash --make_test_from_val 0.2
```

执行后会在 `data/processed` 下生成：

- `images/`
- `labels/`
- `data.yaml`（后续训练所需的配置文件）

---

## 模型训练

### 1. 跑通训练（小规模测试）

```bash
python -m src.train --data data/processed/data.yaml --model yolov8m.pt --imgsz 640 --epochs 3 --batch 16 --workers 16 --device 0 --name stage1_test_640
```

### 2. Stage 1：正式训练（imgsz=640）

```bash
python -m src.train --data data/processed/data.yaml --model yolov8m.pt --imgsz 640 --epochs 150 --batch 16 --workers 16 --device 0 --name stage1_640
```

### 3. Stage 2：微调（imgsz=960）
在640训练的模型基础上进行微调。
```bash
yolo detect train   model=code/runs/detect/stage1_640/weights/best.pt   data=data/processed/data.yaml   imgsz=960   batch=8   epochs=50   device=0   workers=16   cache=disk   amp=True   mosaic=0.0   mixup=0.0   lr0=0.002   patience=15   name=stage2_960_ft
```

---

## 附录：训练辅助工具说明

### 1. TensorBoard 可视化
训练过程中会输出TensorBoard缓存文件logdir
- 启动命令：  
  ```bash
  tensorboard --logdir code/runs/detect/stage1_m_640/ --port 6006 --host 127.0.0.1
  ```

- 若为远程训练，可在本地进行端口转发，例如：  
  ```bash
  ssh -CNg -L 6006:127.0.0.1:6006 root@connect.westb.seetacloud.com -p 33576
  ```
  然后在本地浏览器打开 `http://127.0.0.1:6006/` 查看进度。

### 2. 使用 tmux 保持远程训练不中断
远端训练关闭本地终端会导致训练中止，在本地打开远端终端即可。
- 启动 tmux 会话：  
  ```bash
  tmux new -s yolo
  ```

- 检查是否成功：  
  ```bash
  echo $TMUX
  ```

- 关闭本地终端后训练仍继续，重连命令：  
  ```bash
  tmux attach
  ```

### 3. 训练效果监控

- 会输出每张图像处理速度、准确率
- 推理测试中会生成 PR 曲线、混淆矩阵等指标

---

## 推理测试

完成阶段性训练后，可进行模型性能可视化，超参数可以进行调整，选择最佳即可：  
```bash
python -m src.test   --weights code/runs/detect/stage1_640/weights/best.pt   --data /root/autodl-tmp/data/processed/data.yaml   --split test   --imgsz 640   --conf 0.25   --iou 0.6   --sample_n 50   --out_project code/runs/demo_test   --out_name stage1640
```
