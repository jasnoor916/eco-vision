"""
Smart Recycling Assistant - Orchestration Pipeline
====================================================

Flow per image:
  1. YOLO detects waste objects (category, bbox, confidence)
  2. Each detected object is cropped
  3. A VLM assesses physical condition of the crop (intact/damaged/dirty/food_contaminated)
  4. The (category, condition) pair is looked up in a curated reuse knowledge base
  5. An LLM phrases the retrieved options into one natural suggestion (or declines if none exist)
  6. Per-object results are composed into a structured JSON report + session-level aggregation

Provider strategy (all free tier, no paid usage):
  - Vision call (assess_condition): Gemini (gemini-flash-latest) as primary.
    On any client/server-side failure, automatically falls back to NVIDIA NIM
    (nvidia/nemotron-nano-12b-v2-vl), which has no daily cap (40 RPM).
  - Text-only call (phrase_reuse_suggestion): Groq (openai/gpt-oss-20b).
    Kept on a separate provider entirely so it never competes with the
    vision call for the same rate-limit budget.

Requires:
  pip install ultralytics pillow python-dotenv google-genai groq openai --break-system-packages

Environment (.env file, NOT committed to git):
  GEMINI_API_KEY=...
  GROQ_API_KEY=...
  NIM_API_KEY=...      # from build.nvidia.com, prefix nvapi-
"""

import os
import json
import base64
import time
from io import BytesIO
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from PIL import Image
from ultralytics import YOLO

from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError
import httpx
from groq import Groq
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()  # reads .env in the current working directory into os.environ

YOLO_MODEL_PATH = "../models/recycling_model.pt"  # already-trained weights, lives in repo root's models/ folder
KNOWLEDGE_BASE_PATH = "reuse_knowledge_base.json"
YOLO_CONF_THRESHOLD = 0.25                 # low bar at detection stage; condition/reuse gating happens later
CONDITION_OPTIONS = ["intact", "damaged", "dirty", "food_contaminated"]

GEMINI_VISION_MODEL = "gemini-flash-latest"  # NOTE: pinned versions like "gemini-2.5-flash" get retired and
                                              # start 404ing for new API keys (this happened during testing).
                                              # "-latest" is Google's documented alias that keeps pointing at
                                              # whatever their current flash-tier model is, so this shouldn't
                                              # need updating again as models get replaced.
NIM_VISION_MODEL = "nvidia/nemotron-nano-12b-v2-vl"
GROQ_TEXT_MODEL = "openai/gpt-oss-20b"      # llama-3.3-70b-versatile was deprecated by Groq; this is the current recommended replacement — check console.groq.com/docs/models if this changes

# --- Hard safety cap: no billing is linked to any of the three provider accounts,
# so this cannot turn into a bill. What it DOES protect against is silently burning
# through a day's free-tier quota (e.g. a bug causing YOLO to detect 40 junk objects
# in one blurry frame, each triggering 2 downstream LLM calls). Once this many total
# provider calls have been made in a single script run, everything stops with a clear
# error instead of quietly hammering the API.
MAX_API_CALLS_PER_RUN = 40
_api_call_count = 0


class ApiCallBudgetExceeded(RuntimeError):
    """Raised when MAX_API_CALLS_PER_RUN is hit, to stop the run cleanly."""
    pass


def _register_api_call(label: str) -> None:
    global _api_call_count
    _api_call_count += 1
    if _api_call_count > MAX_API_CALLS_PER_RUN:
        raise ApiCallBudgetExceeded(
            f"Stopped after {MAX_API_CALLS_PER_RUN} provider calls this run "
            f"(hit while attempting: {label}). Raise MAX_API_CALLS_PER_RUN in the "
            f"config section if this run genuinely needs more, and check the YOLO "
            f"detection count first if this trips unexpectedly — a large number of "
            f"detections on one image is the most likely cause."
        )

# --- Provider clients (constructed once, reused across calls) ---

_gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

_nim_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NIM_API_KEY"],
)

_groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ConditionResult:
    status: str
    justification: str
    assessment_confidence: float
    provider_used: str   # "gemini" or "nim" — useful for debugging/eval logging


@dataclass
class ReuseResult:
    eligible: bool
    reason: Optional[str] = None
    title: Optional[str] = None
    why: Optional[str] = None
    steps: Optional[list[str]] = None
    source_citation: Optional[str] = None

@dataclass
class ObjectResult:
    object_id: str
    bbox: list
    category: str
    yolo_confidence: float
    condition: ConditionResult
    contamination_flag: bool
    reuse: ReuseResult

    def to_dict(self):
        return {
            "object_id": self.object_id,
            "bbox": self.bbox,
            "category": self.category,
            "yolo_confidence": round(self.yolo_confidence, 3),
            "condition": {
                "status": self.condition.status,
                "justification": self.condition.justification,
                "assessment_confidence": round(self.condition.assessment_confidence, 3),
                "provider_used": self.condition.provider_used,
            },
            "contamination_flag": self.contamination_flag,
            "reuse": {
                "eligible": self.reuse.eligible,
                "reason": self.reuse.reason,
                "title": self.reuse.title,
                "why": self.reuse.why,
                "steps": self.reuse.steps,
                "source_citation": self.reuse.source_citation,
            },
        }


# ---------------------------------------------------------------------------
# Step 1: YOLO detection
# ---------------------------------------------------------------------------

def run_detection(image_path: str, model: YOLO) -> list[dict]:
    """Returns a list of {bbox, category, confidence} dicts."""
    results = model(image_path, conf=YOLO_CONF_THRESHOLD)[0]
    detections = []
    for i, box in enumerate(results.boxes):
        cls_id = int(box.cls.item())
        category = results.names[cls_id]
        conf = float(box.conf.item())
        xyxy = [round(v, 1) for v in box.xyxy[0].tolist()]
        detections.append({
            "object_id": f"obj_{i:03d}",
            "bbox": xyxy,
            "category": category,
            "confidence": conf,
        })
    return detections


def crop_object(image: Image.Image, bbox: list) -> Image.Image:
    x1, y1, x2, y2 = bbox
    return image.crop((x1, y1, x2, y2))


def image_to_base64(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="JPEG")
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Step 2: Condition assessment (VLM call, isolated from reuse reasoning)
# Gemini primary -> NIM fallback on rate limit
# ---------------------------------------------------------------------------

def _condition_prompt(category: str) -> str:
    return f"""You are assessing the physical condition of a single waste item shown in this image crop.
The item has been classified as: {category}

Respond ONLY with JSON in this exact format:
{{
  "status": one of {CONDITION_OPTIONS},
  "justification": "one short sentence describing what you observed",
  "assessment_confidence": a float between 0 and 1
}}

Do not suggest reuse ideas here. Only assess physical condition.
Base "assessment_confidence" on image clarity and how unambiguous the visual evidence is -
lower confidence if the crop is blurry, partially occluded, or damage is uncertain."""


def _assess_condition_gemini(crop: Image.Image, category: str) -> dict:
    _register_api_call("Gemini condition assessment")
    prompt = _condition_prompt(category)
    buf = BytesIO()
    crop.save(buf, format="JPEG")
    image_part = types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")
    response = _gemini_client.models.generate_content(
        model=GEMINI_VISION_MODEL,
        contents=[prompt, image_part],
        config=types.GenerateContentConfig(
            temperature=0,
        ),
    )
    return _parse_json_response(response.text)   

def _assess_condition_nim(crop: Image.Image, category: str) -> dict:
    _register_api_call("NIM condition assessment (fallback)")
    prompt = _condition_prompt(category)
    img_b64 = image_to_base64(crop)
    response = _nim_client.chat.completions.create(
        model=NIM_VISION_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            ],
        }],
        max_tokens=300,
        temperature=0,
    )
    return _parse_json_response(response.choices[0].message.content)


def assess_condition(crop: Image.Image, category: str) -> ConditionResult:
    provider_used = "gemini"
    try:
        parsed = _assess_condition_gemini(crop, category)
    except (ClientError, ServerError, httpx.TransportError) as e:
        # Covers 429 rate-limit, 404 model-retired, and any 5xx from Gemini's side.
        # Rather than branch on the exact status code, any client/server-side failure
        # from Gemini falls through to NIM, which is a fully independent free provider.
        print(f"[assess_condition] Gemini call failed ({e}); falling back to NIM")
        provider_used = "nim"
        time.sleep(1)  # brief pause before switching providers
        parsed = _assess_condition_nim(crop, category)

    return ConditionResult(
        status=parsed.get("status", "damaged"),   # fail closed: assume damaged if parsing fails
        justification=parsed.get("justification", "assessment unavailable"),
        assessment_confidence=float(parsed.get("assessment_confidence", 0.0)),
        provider_used=provider_used,
    )


# ---------------------------------------------------------------------------
# Step 3: Reuse retrieval (dict lookup, not model-generated)
# ---------------------------------------------------------------------------

def load_knowledge_base(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def retrieve_reuse_options(kb: dict, category: str, condition_status: str) -> list[dict]:
    return kb.get(category, {}).get(condition_status, [])


# ---------------------------------------------------------------------------
# Step 4: Reuse phrasing (LLM selects/phrases from retrieved options only)
# Groq, text-only
# ---------------------------------------------------------------------------

def phrase_reuse_suggestion(category: str, condition_status: str, retrieved_entries: list[dict]) -> ReuseResult:
    if not retrieved_entries:
        return ReuseResult(
            eligible=False,
            reason=f"no verified reuse option exists for {category} in condition '{condition_status}'",
        )

    _register_api_call("Groq reuse phrasing")

    prompt = f"""
You are an AI assistant helping users reuse household waste responsibly.

Waste item:
- Category: {category}
- Condition: {condition_status}

Below are VERIFIED reuse options retrieved from a trusted knowledge base.
Only use the information provided. Do NOT invent new reuse ideas.

Knowledge Base:
{json.dumps(retrieved_entries, indent=2)}

Return ONLY valid JSON in this exact format:

{{
  "title": "short title (4-8 words)",
  "why": "One sentence explaining why this item can or cannot be reused.",
  "steps": [
    "Practical reuse idea 1",
    "Practical reuse idea 2",
    "Practical reuse idea 3"
  ],
  "source_citation": "source from the selected knowledge base entry"
}}

Rules:
- Use simple everyday English.
- Keep each step under 15 words.
- If fewer than three ideas exist, return only those.
- Never invent reuse ideas outside the supplied knowledge base.
- Return JSON only.
"""

    response = _groq_client.chat.completions.create(
        model=GROQ_TEXT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=200,
        reasoning_effort="low",
    )

    choice = response.choices[0]

    print("\n===== GROQ DEBUG =====")
    print("finish_reason:", choice.finish_reason)
    print("message:", choice.message)
    print("======================\n")

    raw_text = choice.message.content or ""
    if not raw_text.strip():
        return ReuseResult(
        eligible=False,
        reason="Groq returned an empty response."
    )
    parsed = _parse_json_response(raw_text)
    if not parsed.get("suggestion"):
        print(f"[phrase_reuse_suggestion] Failed to parse Groq response as JSON. Raw text was:\n{raw_text!r}")
        return ReuseResult(
            eligible=False,
            reason="reuse phrasing call failed to return valid JSON (see logs)",
        )

    return ReuseResult(
        eligible=True,
        title=parsed.get("title"),
        why=parsed.get("why"),
        steps=parsed.get("steps", []),
        source_citation=parsed.get("source_citation"),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json_response(text: str) -> dict:
    text = (text or "").strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        print(f"[_parse_json_response] Response was not valid JSON. Raw text was:\n{text!r}")
        return {}

    if not isinstance(result, dict):
        print(f"[_parse_json_response] Parsed JSON was not a dict (got {type(result).__name__}). Raw text was:\n{text!r}")
        return {}

    return result


CONTAMINATION_STATUSES = {"dirty", "food_contaminated"}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_image(image_path: str, yolo_model: YOLO, kb: dict) -> dict:
    image = Image.open(image_path).convert("RGB")
    detections = run_detection(image_path, yolo_model)

    objects: list[ObjectResult] = []
    for det in detections:
        crop = crop_object(image, det["bbox"])
        condition = assess_condition(crop, det["category"])
        retrieved = retrieve_reuse_options(kb, det["category"], condition.status)
        reuse = phrase_reuse_suggestion(det["category"], condition.status, retrieved)

        objects.append(ObjectResult(
            object_id=det["object_id"],
            bbox=det["bbox"],
            category=det["category"],
            yolo_confidence=det["confidence"],
            condition=condition,
            contamination_flag=condition.status in CONTAMINATION_STATUSES,
            reuse=reuse,
        ))

    session_summary = {
        "total_objects": len(objects),
        "category_breakdown": _count_by(objects, lambda o: o.category),
        "contamination_count": sum(o.contamination_flag for o in objects),
        "reuse_eligible_count": sum(o.reuse.eligible for o in objects),
    }

    return {
        "objects": [o.to_dict() for o in objects],
        "session_summary": session_summary,
    }


def _count_by(objects: list[ObjectResult], key_fn) -> dict:
    counts: dict = {}
    for o in objects:
        k = key_fn(o)
        counts[k] = counts.get(k, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    yolo_model = YOLO(YOLO_MODEL_PATH)
    kb = load_knowledge_base(KNOWLEDGE_BASE_PATH)

    try:
        result = process_image("test_image.jpg", yolo_model, kb)
        print(json.dumps(result, indent=2))
    except ApiCallBudgetExceeded as e:
        print(f"\n[STOPPED] {e}")
    finally:
        print(f"\nTotal provider calls this run: {_api_call_count}/{MAX_API_CALLS_PER_RUN}")