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
<<<<<<< HEAD
=======
# ♻️ EcoVision: Smart Recycling Assistant

An AI-powered, retrieval-grounded recycling assistant that combines **computer vision**, **multimodal AI**, and **knowledge-grounded reasoning** to provide safe, explainable reuse recommendations for household waste.

Developed as the final project for **M155.006900 – First Steps in Programming a Humanoid AI Robot** at **Seoul National University (SNU)**.

---

## Project Overview

EcoVision extends a custom **YOLOv8** waste detection model with a condition-aware decision pipeline capable of determining whether detected waste items are suitable for reuse before generating recommendations.

Unlike conventional waste classifiers that only identify object categories, EcoVision performs **visual condition assessment**, retrieves verified reuse knowledge from a curated database, and generates human-readable recommendations while preventing unsupported or unsafe advice.

The project was developed within Seoul National University's hands-on humanoid robotics course, which covers robotics, computer vision, deep learning, custom object detection, and AI-driven robotic perception. :contentReference[oaicite:1]{index=1}

---

## Key Features

- YOLOv8-based waste object detection
- Multimodal visual condition assessment
- Retrieval-grounded reuse recommendations
- Knowledge-base constrained LLM generation
- Structured per-object JSON reporting
- Automatic session-level analytics
- Multi-provider AI pipeline with automatic fallback
- Free-tier API budgeting for reproducible deployment

---

## System Architecture

```
                Input Image
                     │
                     ▼
          YOLOv8 Object Detection
                     │
                     ▼
           Crop Detected Objects
                     │
                     ▼
      Visual Condition Assessment
      (Gemini → NVIDIA NIM Fallback)
                     │
                     ▼
      Retrieve Matching Reuse Knowledge
                     │
                     ▼
      Groq LLM Formats Recommendation
                     │
                     ▼
        Structured JSON Output
                     │
                     ▼
        Session-Level Aggregation
```

---

## Motivation

Most waste classification systems stop after identifying an object's category.

However, whether an item **can actually be reused** depends on its physical condition.

For example:

| Object | Condition | Recommendation |
|---------|-----------|----------------|
| Paper | Intact | Reuse possible |
| Paper | Food contaminated | Recycle |
| Plastic bottle | Clean | Reuse possible |
| Broken glass | Broken | No recommendation |

EcoVision explicitly separates:

- **Visual understanding** (condition assessment)
- **Reuse reasoning** (knowledge retrieval)
- **Natural-language explanation** (LLM formatting)

This modular design improves explainability and prevents unsafe reuse suggestions.

---

## Technology Stack

### Computer Vision

- YOLOv8
- OpenCV
- Pillow

### AI Models

- Google Gemini Flash
- NVIDIA NIM Vision
- Groq GPT-OSS-20B

### Backend

- Python
- JSON Knowledge Base
- dotenv

---

## Repository Structure

```
eco-vision/

├── data/
│   ├── train/
│   ├── valid/
│   └── test/
│
├── models/
│   └── recycling_model.pt
│
├── reuse_pipeline/
│   ├── waste_pipeline.py
│   ├── run_batch.py
│   ├── reuse_knowledge_base.json
│   └── build_eval_template.py
│
├── predict_test_images.py
├── train.ipynb
├── requirements.txt
└── README.md
```

---

## Example Output

```json
{
  "category": "Paper",
  "condition": "intact",
  "reuse": {
    "eligible": true,
    "title": "Reuse Paper for Everyday Tasks",
    "why": "The paper is clean and undamaged, making it suitable for reuse.",
    "steps": [
      "Use as scratch paper.",
      "Use as notebook filler.",
      "Use for packaging cushioning."
    ]
  }
}
```

---

## Design Principles

### Condition-Aware Decision Making

Reuse recommendations are generated only after independently assessing the physical condition of each detected object.

---

### Retrieval-Grounded Generation

The language model **never invents reuse ideas**.

Recommendations are selected exclusively from a curated reuse knowledge base.

---

### Safety by Design

Unsafe category-condition combinations intentionally return **no reuse recommendation** rather than hallucinated advice.

---

### Explainable Outputs

Every detected object is represented as structured JSON containing:

- detection result
- condition assessment
- reuse eligibility
- explanation
- reusable actions
- source citation

---

## Future Work

- OCR-assisted waste identification
- Robotic pick-and-place integration
- Mobile deployment
- Expanded reuse knowledge base
- Multi-language recommendations
- Confidence calibration for condition assessment

---

## Course Context

This project was completed as part of:

**M155.006900 – First Steps in Programming a Humanoid AI Robot**

Seoul National University

The course provides hands-on experience in:

- Robotics
- Computer Vision
- Machine Learning
- Deep Learning
- Custom Object Detection
- AI for Humanoid Robots

The final assessment is centered on a team-based robotics and AI project, allowing students to design and implement an intelligent robotic capability. :contentReference[oaicite:2]{index=2}

---

## Author

**Jasnoor Kaur**

B.S. Chemistry, IIT Bombay

Seoul National University International Summer Program 2026
>>>>>>> 629ace58 (Initial commit: EcoVision Smart Recycling Assistant)
