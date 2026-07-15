# eco-vision

A YOLO object detector that draws bounding boxes around five waste categories —
**Glass, Metal, Paper, Plastic, Waste** — for sorting/recycling.

Everything runs from one notebook, cross-platform (macOS with the MPS GPU, or
Windows/Linux on CPU or an NVIDIA GPU).

## Dataset

Already in the standard Ultralytics layout under `data/`:

```
data/
  data.yaml                 # class names (nc=5) + split paths
  train/{images,labels}/    # 3502 images
  valid/{images,labels}/    #  580 images
  test/{images,labels}/     #   45 images
```

## Setup

```bash
# from the project root
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt jupyter ipykernel
```

On **Windows with an NVIDIA GPU**, after the install above, run the optional
CUDA-torch line inside the notebook's install cell to enable GPU training.

## Run

Open the notebook and run the cells top to bottom:

```bash
./.venv/bin/jupyter notebook notebooks/train_and_predict.ipynb
```

The notebook:

1. Installs dependencies (once per environment).
2. Auto-detects the compute device (CUDA → MPS → CPU).
3. Resolves an absolute `data.yaml` so images always load.
4. Exposes training parameters in one editable cell.
5. Trains `yolo11n` with early stopping → writes `runs/waste/weights/best.pt`.
6. Validates and prints mAP on the validation split.
7. Runs prediction on a test image and shows the boxes inline.

### Tuning

Edit the **Parameters** cell: `MODEL` (e.g. `yolov10s`, `yolo11m`), `EPOCHS`,
`BATCH`, `IMGSZ`, `PATIENCE`, `CONF`. Trained weights land in
`runs/waste/weights/best.pt`.

## Notes

- Training on a laptop CPU is slow; expect meaningfully faster runs on Apple
  Silicon (MPS) or an NVIDIA GPU (CUDA).
- `data/data.resolved.yaml`, `runs/`, `.venv/`, and `*.pt` are generated
  artifacts and are git-ignored.
