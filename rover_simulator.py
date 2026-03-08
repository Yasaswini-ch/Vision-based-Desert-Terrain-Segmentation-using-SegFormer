import asyncio
import json
import os
import time
import random
from pathlib import Path
import websockets
import httpx
from PIL import Image
import io

# Config
SERVER_WS_URL = f"ws://{os.getenv('HOST_IP', '0.0.0.0')}:8001/ws/rover"
# Exact path from user request
# Use pathlib for cross-platform backslash handling
TEST_IMAGES_DIR = Path(r"C:\Opensource\desert segmentation\Offroad_Segmentation_testImages\Color_Images")
LOG_FILE = "mission_log.json"

class RoverSimulator:
    def __init__(self):
        self.mission_log = []
        self.frames_sent = 0
        self.total_frames = 0
        self.is_paused = False
        self.delay = 2.0  # Default 2s
        self.images = []

    def load_images(self):
        print("═" * 32)
        print(f"Scanning folder: {TEST_IMAGES_DIR}")
        
        if not TEST_IMAGES_DIR.exists():
            print(f"❌ ERROR: Test images directory NOT FOUND at:")
            print(f"   {TEST_IMAGES_DIR}")
            print("═" * 32)
            return False
            
        # Scan for PNG and JPG
        self.images = sorted(list(TEST_IMAGES_DIR.glob("*.png")) + list(TEST_IMAGES_DIR.glob("*.jpg")))
        self.total_frames = len(self.images)
        
        if self.total_frames == 0:
            print(f"❌ ERROR: No images found in {TEST_IMAGES_DIR}")
            print("═" * 32)
            return False
            
        print(f"✅ Found {self.total_frames} images.")
        print("\nFirst 5 images found:")
        for img in self.images[:5]:
            print(f"  - {img.name}")
        print("═" * 32)
        return True

    async def run(self):
        if not self.load_images():
            return

        # Shuffle for live unpredictable feed
        random.shuffle(self.images)
        print(f"🔀 Shuffled {self.total_frames} images for live demo.")
        print(f"Connecting to {SERVER_WS_URL}...")
        
        try:
            async with websockets.connect(SERVER_WS_URL, max_size=10 * 1024 * 1024) as ws:
                print("✅ Connected to Inference Server\n")
                
                idx = 0
                while True: # Loop indefinitely
                    if self.is_paused:
                        await asyncio.sleep(0.5)
                        continue

                    img_path = self.images[idx % self.total_frames]
                    try:
                        with open(img_path, "rb") as f:
                            image_data = f.read()
                        
                        start_time = time.perf_counter()
                        await ws.send(image_data)
                        
                        # Wait for command
                        response = await ws.recv()
                        command = json.loads(response)
                        latency = int((time.perf_counter() - start_time) * 1000)
                        
                        self.print_status(idx % self.total_frames + 1, img_path.name, command, latency)
                        
                        await asyncio.sleep(self.delay) 
                        idx += 1
                        
                    except Exception as e:
                        print(f"Error processing frame {img_path.name}: {e}")
                        await asyncio.sleep(1)
                        
        except Exception as e:
            print(f"Connection failed: {e}")

    def print_status(self, count, filename, cmd, latency):
        trav = cmd.get("traversable_pct", 0.0)
        heading = cmd.get("heading", "STOP").replace("_", " ").upper()
        speed = cmd.get("speed_kmh", 0.0)
        risk = cmd.get("risk_level", "LOW")
        hazards = cmd.get("hazards", [])
        
        print("════════════════════════════════")
        print(f"[ROVER CAM] Frame {count:03d}/{self.total_frames}")
        print(f"[ROVER CAM] Image: {filename}")
        print(f"[SEG]       Processing...")
        print(f"[SEG]       Done in {latency}ms")
        print(f"[NAV]       Traversable: {trav:.1f}%")
        print(f"[NAV]       Command: {heading}")
        print(f"[NAV]       Speed: {speed} km/h")
        print(f"[NAV]       Risk: {risk}")
        if hazards:
            h_label = hazards[0].replace("_", " ").upper()
            print(f"[HAZARD]    ⚠️  {h_label}")
        print("════════════════════════════════")

if __name__ == "__main__":
    sim = RoverSimulator()
    try:
        asyncio.run(sim.run())
    except KeyboardInterrupt:
        print("\nSimulation stopped.")

if __name__ == "__main__":
    sim = RoverSimulator()
    asyncio.run(sim.run())
