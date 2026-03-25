# ml_ludy_obuchenie

YOLO26 training on Windows

## Setup

PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Command Prompt:

```bat
py -m venv .venv
.\.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Train on Kaggle dataset

PowerShell:

```powershell
python .\train_yolo26.py `
  --dataset fareselmenshawii/human-dataset `
  --model yolo26n.pt `
  --epochs 100 `
  --imgsz 640 `
  --batch 16 `
  --device cpu
```

Command Prompt:

```bat
python train_yolo26.py ^
  --dataset fareselmenshawii/human-dataset ^
  --model yolo26n.pt ^
  --epochs 100 ^
  --imgsz 640 ^
  --batch 16 ^
  --device cpu
```

Notes:

- For NVIDIA GPU, usually use `--device 0`.
- For CPU, use `--device cpu`.
- `mps` is only for macOS, not Windows.
- If the dataset is already in YOLO format, the script will generate `data.yaml` automatically.
- COCO and Pascal VOC detection layouts are also supported.
