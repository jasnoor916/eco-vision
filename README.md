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
