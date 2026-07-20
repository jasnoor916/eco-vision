"""
Gretchen's action layer: robot gestures + Groq recycling advice + TTS.

This is adapted from the existing suggest.py, with two changes:
  1. The Groq API key now comes from an environment variable, not a
     hardcoded string. Set it before running:
         export GROQ_API_KEY="your_new_key_here"
     (rotate the old key — it was pasted into a chat, treat it as leaked.)
  2. get_recycling_advice() no longer assumes the item is specifically a
     "bottle" or "cup" — it takes whatever label the vision pipeline hands
     it (e.g. "Plastic", "Paper", "General", or a specific object name like
     "plastic bottle" if your detector gives you that granularity).

trigger() is the single entry point the grip-detection pipeline calls,
exactly once per confirmed grip event.
"""

import asyncio
import os
import textwrap
import time

import cv2
import edge_tts
from groq import Groq

from gretchen.robot import Robot

VOICE = "en-GB-SoniaNeural"
COOLDOWN = 10  # seconds — minimum gap between advice calls for the same label

GROQ_API_KEY = "GROQ_API_KEY_REDACTED"
client = Groq(api_key=GROQ_API_KEY)

robot = Robot('COM3', 0)
robot.start()
robot.start_motors()

_last_label = None
_last_call_time = 0.0
display_text = "Show me something to recycle!"


def nod():
    for _ in range(3):
        robot.up(); time.sleep(0.3)
        robot.down(); time.sleep(0.3)
    robot.up()


def shake():
    for _ in range(3):
        robot.left(); time.sleep(0.3)
        robot.right(); time.sleep(0.3)
    robot.left()


def draw_text(img, text):
    max_width_chars = 40
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    color = (0, 255, 0)
    line_spacing = 35

    lines = textwrap.wrap(text, width=max_width_chars)
    box_height = (len(lines) * line_spacing) + 20
    cv2.rectangle(img, (0, 0), (640, box_height), (0, 0, 0), -1)

    y = 40
    for line in lines:
        cv2.putText(img, line, (20, y), font, font_scale, color, thickness)
        y += line_spacing
    return img


def get_recycling_advice(item_label: str) -> str:
    system_prompt = (
        "You are Gretchen, a recycling expert in South Korea. "
        "CRITICAL RULES: "
        "1. For plastic bottles: ALWAYS start with 'YES'. "
        "2. For disposable cups: ALWAYS start with 'NO'. "
        "3. For other plastic, paper, or general waste items, decide based "
        "on standard South Korean recycling rules. "
        "4. Start your response strictly with 'YES' or 'NO' followed by a "
        "short reason or reuse tip."
    )
    try:
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": (
                    f"Is a {item_label} recyclable in South Korea or not? "
                    "Provide a short reason."
                )},
            ],
            model="llama-3.3-70b-versatile",
        )
        response = completion.choices[0].message.content

        if response.strip().upper().startswith("YES"):
            print(f"Gretchen (Nodding): {response}")
            nod()
        else:
            print(f"Gretchen (Shaking): {response}")
            shake()

        return response
    except Exception as e:
        return f"System error: {e}"


async def _speak_async(text: str):
    communicate = edge_tts.Communicate(text, VOICE)
    await communicate.save("output.mp3")
    os.system("afplay output.mp3")  # macOS; swap for `mpg123`/`ffplay` on Linux


def trigger(category: str, confidence: float) -> str:
    """
    Called once per confirmed grip event by the hand-landmark pipeline.
    Returns the display text so the caller can draw it on-screen.
    """
    global _last_label, _last_call_time, display_text

    now = time.time()
    if category == _last_label and (now - _last_call_time) < COOLDOWN:
        return display_text  # same item, still in cooldown — skip repeat call

    display_text = f"Analyzing {category} (conf {confidence:.2f})..."
    print(display_text)

    advice = get_recycling_advice(category)
    display_text = advice
    asyncio.run(_speak_async(advice))

    _last_label = category
    _last_call_time = now
    return display_text
