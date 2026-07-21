"""
Gretchen's action layer: robot gestures + Groq recycling advice + TTS.

This is adapted from the existing suggest.py, with two changes:
  1. The Groq API key, motor port, and camera now come from the .env file
     at the project root, not hardcoded strings. Edit .env to match your
     machine. (rotate the old key — it was pasted into a chat, treat it as
     leaked.)
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
from pathlib import Path

import cv2
import edge_tts
from groq import Groq
from dotenv import load_dotenv

# Load settings from the .env at the repo root (this file lives at the root, so
# .env sits beside it). See .env for all options and per-OS values.
load_dotenv(Path(__file__).resolve().parent / ".env")


def _camera(value):
    """A plain number is a camera index (int); anything else is a device path."""
    return int(value) if value.isdigit() else value


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ROBOT_MOTOR_PORT = os.getenv("ROBOT_MOTOR_PORT", "/dev/tty.usbserial-FT94ELHH")
ROBOT_CAMERA = _camera(os.getenv("ROBOT_CAMERA", "0"))

VOICE = "en-GB-SoniaNeural"
COOLDOWN = 10  # seconds — minimum gap between advice calls for the same label

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

robot = None


def start_robot(motor_port=ROBOT_MOTOR_PORT, camera=ROBOT_CAMERA):
    """Initialize the robot only when the main program selects robot mode."""
    global robot
    from gretchen.robot import Robot

    robot = Robot(motor_port, camera)
    robot.start()
    return robot


def stop_robot():
    """Release the robot camera and motor connection if they were started."""
    global robot
    if robot is None:
        return
    if robot.camera.vc is not None:
        robot.camera.vc.release()
    robot.disconnect()
    robot = None

_last_label = None
_last_call_time = 0.0
display_text = "Show me something to recycle!"


def nod():
    if robot is None:
        print("Robot disabled: skipping nod gesture.")
        return
    for _ in range(3):
        robot.up()
        time.sleep(0.3)
        robot.down()
        time.sleep(0.3)
    robot.up()


def shake():
    if robot is None:
        print("Robot disabled: skipping shake gesture.")
        return
    for _ in range(3):
        robot.left()
        time.sleep(0.3)
        robot.right()
        time.sleep(0.3)
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
    if client is None:
        return "System error: GROQ_API_KEY is not set in the project .env file."

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
            model="openai/gpt-oss-120b",
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
    # macOS; swap for `mpg123`/`ffplay` on Linux
    os.system("afplay output.mp3")


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
