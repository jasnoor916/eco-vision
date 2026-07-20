import cv2
import time
from gretchen.robot import Robot
from ultralytics import YOLO
from groq import Groq
import textwrap
import pyttsx3
import asyncio
import edge_tts
import os
# Initialize the engine
engine = pyttsx3.init()

# TTS Configuration
VOICE = "en-GB-SoniaNeural"

# Initialize
client = Groq(api_key="GROQ_API_KEY_REDACTED")
robot = Robot('/dev/tty.usbserial-FT94EO15', 0)
robot.start()
robot.start_motors()

model = YOLO('yolov8n.pt')
# Restricted to 39: bottle, 41: cup
TARGET_CLASSES = [39, 41] 
CONFIDENCE_THRESHOLD = 0.60
COOLDOWN = 10 
last_detected = None
last_call_time = 0
display_text = "Show me a bottle or cup!"

def draw_text(img, text):
    """Wraps text and draws it on a clean background."""
    # Settings for visual appearance
    max_width_chars = 40 
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    color = (0, 255, 0) # Green text
    line_spacing = 35
    
    # Wrap text into lines
    lines = textwrap.wrap(text, width=max_width_chars)
    
    # Draw background box scaled to text height
    box_height = (len(lines) * line_spacing) + 20
    cv2.rectangle(img, (0, 0), (640, box_height), (0, 0, 0), -1)
    
    # Draw each line
    y = 40
    for line in lines:
        cv2.putText(img, line, (20, y), font, font_scale, color, thickness)
        y += line_spacing
    return img

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

def get_recycling_advice(item_name):
    system_prompt = (
        "You are Gretchen, a recycling expert in South Korea. "
        "CRITICAL RULES: "
        "1. For 'plastic bottle' or 'bottle': ALWAYS start with 'YES'. "
        "2. For 'cup' (especially disposable plastic cups): ALWAYS start with 'NO'. "
        "3. Start your response strictly with 'YES' or 'NO' followed by how to resuse it. "
    )
    try:
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Is a {item_name} recyclable in South Korea or not? Provide a short reason."}
            ],
            model="llama-3.3-70b-versatile"
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


async def speak_async(text):
    """Generates and plays high-quality audio."""
    communicate = edge_tts.Communicate(text, VOICE)
    await communicate.save("output.mp3")
    os.system("afplay output.mp3")

def main():
    global last_detected, last_call_time, display_text
    cv2.namedWindow("Gretchen Vision")

    while True:
        ret, img, timestamp = robot.camera.getImage()
        if not ret: continue

        # Only detect bottles and cups
        results = model.predict(img, classes=TARGET_CLASSES, conf=CONFIDENCE_THRESHOLD, verbose=False)

        for r in results:
            for box in r.boxes:
                label = model.names[int(box.cls[0])]
                if label != last_detected or (time.time() - last_call_time) > COOLDOWN:
                    display_text = f"Analyzing {label}..."
                    display_text = get_recycling_advice(label)
                    asyncio.run(speak_async(display_text))
                    last_detected = label
                    last_call_time = time.time()

        # Display detection box + text advice
        display_img = results[0].plot() if len(results) > 0 else img
        display_img = draw_text(display_img, display_text)
        
        cv2.imshow("Gretchen Vision", display_img)
        if cv2.waitKey(1) > 0: break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()