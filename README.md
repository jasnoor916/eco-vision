# ♻️ EcoVision: Smart Recycling Assistant

A YOLO waste detector wired to the Gretchen humanoid robot. The detector draws
bounding boxes around five waste categories — **Glass, Metal, Paper, Plastic,
Waste** — and the robot responds to what it sees with spoken recycling advice.

Final project for **M155.006900 – First Steps in Programming a Humanoid AI
Robot**, Seoul National University International Summer Program 2026.

---

## Quick start

Everything installs with one command. Run these from the project root:

```bash
git clone https://teaching.csap.snu.ac.kr/hafsahnasir/eco-vision.git
cd eco-vision

python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

That single `pip install` also builds the two vendored robot libraries
(`gretchen` and `dynamixel_sdk`) — they are listed as editable entries at the
bottom of `requirements.txt`, so there is no separate install step.

Confirm it worked:

```bash
./.venv/bin/python -c "from gretchen.robot import Robot; print('robot lib OK')"
./.venv/bin/python predict_test_images.py --count 3
```

The second command runs the trained detector on three test images and writes
annotated copies to `runs/test_predictions/`. **No robot hardware needed** — this
is the fastest way to confirm the whole setup works.


---

## Repository structure

```
eco-vision/
├── data/                        # dataset, Ultralytics layout
│   ├── data.yaml                #   5 classes + split paths
│   ├── train/{images,labels}/   #   3502 images
│   ├── valid/{images,labels}/   #    580 images
│   └── test/{images,labels}/    #     45 images
│
├── models/
│   └── best.pt                  # trained waste detector
│
├── project/                     # robot-side code + vendored libraries
│   ├── suggest.py               #   robot demo: detect → advise → speak → nod
│   ├── yolov8n.pt               #   stock COCO weights used by the demo
│   ├── gretchen/                #   course robot library  (installed editable)
│   │   └── course_material/lib/ #     ← the actual Python package lives here
│   └── lib/DynamixelSDK/        #   motor SDK             (installed editable)
│       └── python/              #     ← the actual Python package lives here
│
├── train.py                     # train the detector on data/
├── inference.py                 # live detection from the robot camera
├── predict_test_images.py       # batch predictions on the test split
├── requirements.txt             # all dependencies, including the two above
└── README.md
```

Note that `project/gretchen/` and `project/lib/DynamixelSDK/` are the upstream
*repositories*, vendored in full. The importable Python packages sit several
levels inside them, which is why they are installed rather than imported by
path. `requirements.txt` handles this for you.

---

## Running

### Batch predictions — no hardware required

```bash
python predict_test_images.py --count 10
```

| Flag | Default | Meaning |
|---|---|---|
| `--count` | 10 | how many test images to sample |
| `--conf` | 0.25 | confidence threshold |
| `--seed` | 42 | sampling seed, for reproducible picks |
| `--model` | `models/best.pt` | weights to load |
| `--source` | `data/test/images` | image folder |
| `--output` | `runs/test_predictions` | where annotated images go |

### Training

```bash
python train.py
```

Selects CUDA on an NVIDIA machine, MPS on Apple Silicon, and CPU otherwise.
Weights are written under `runs/`. CPU training is slow — expect meaningfully
faster runs on MPS or CUDA.

### Robot demo — requires hardware

```bash
python project/suggest.py
```

Detects bottles and cups from the robot camera, asks a Groq-hosted Llama model
whether the item is recyclable in South Korea, speaks the answer, and nods or
shakes accordingly.

The Groq API key and the robot serial port are set at the top of
`project/suggest.py`. Change the port to match your machine.

### Main EcoVision pipeline

Run the full hand, grip, item-detection, recycling-advice, and speech pipeline
using only the laptop camera (no robot or motor initialization):

```bash
python 4_main.py --mode laptop
```

Run the same pipeline with Gretchen's camera, motors, and gestures:

```bash
python 4_main.py --mode robot
```

Use `--camera 1` to select a different laptop camera. In robot mode,
`--camera` and `--motor-port` override the values from `.env`.

### Live camera detection

```bash
# Laptop camera only (does not initialize the robot or motors)
python inference.py --mode laptop

# Robot camera and motors
python inference.py --mode robot
```

Both modes accept a camera index or device path. Robot mode also accepts a
motor serial port:

```bash
python inference.py --mode laptop --camera 1
python inference.py --mode robot --camera 0 --motor-port /dev/tty.usbserial-FT94ELHH

# Show every available option
python inference.py --help
```

Robot mode remains the default, so the original command still works:

```bash
python inference.py
```

---

## Dataset

`data/data.yaml` deliberately omits a `path:` key, so Ultralytics resolves the
split folders relative to the YAML's own location. The dataset therefore works
from any working directory on any OS.

| id | Class |
|---|---|
| 0 | Glass |
| 1 | Metal |
| 2 | Paper |
| 3 | Plastic |
| 4 | Waste |

---

## Platform notes

**Serial port.** The robot port differs per machine — macOS uses
`/dev/tty.usbserial-FT94EO15`, Windows uses `COM3`. Set it at the top of
`project/suggest.py` and on line 7 of `inference.py`.

**Audio playback.** `project/suggest.py` plays speech with `afplay`, which is
macOS-only. On Windows or Linux, replace that call with `start` or `aplay`.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'gretchen'`**
The editable install did not run. Re-run `pip install -r requirements.txt` from
the project root — the `-e` paths in it are relative, so the working directory
matters.

**VS Code underlines `from gretchen.robot import Robot` even though it runs**
Pylance resolves imports against the selected interpreter, not the folder
layout. **Cmd+Shift+P → Python: Select Interpreter → `./.venv/bin/python`**,
then reload the window.

**`termios.error: (19, 'Operation not supported by device')`**
`gretchen/motors.py` reads terminal settings when the module is imported, so it
needs a real terminal. This appears when running under a Jupyter kernel, CI, or
a piped shell — run from an interactive terminal instead.
