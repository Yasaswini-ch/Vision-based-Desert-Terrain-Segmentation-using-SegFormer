import os
import time
import json
import httpx
from pathlib import Path
from datetime import datetime

# Configuration
FALCONSIM_INPUT_DIR  = Path("C:/FalconSim/output/camera/")
FALCONSIM_OUTPUT_DIR = Path("C:/FalconSim/input/commands/")
API_URL = "http://localhost:8000/segment"
POLL_INTERVAL_MS = 200
INTEGRATION_MODE = "falconsim" 

def ensure_dirs():
    """Ensure mock directories exist for demo if real ones don't."""
    if not FALCONSIM_INPUT_DIR.exists():
        os.makedirs(FALCONSIM_INPUT_DIR, exist_ok=True)
        print(f"Created mock input dir: {FALCONSIM_INPUT_DIR}")
    if not FALCONSIM_OUTPUT_DIR.exists():
        os.makedirs(FALCONSIM_OUTPUT_DIR, exist_ok=True)
        print(f"Created mock output dir: {FALCONSIM_OUTPUT_DIR}")

def process_frame(file_path):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Processing: {file_path.name}")
    
    try:
        with open(file_path, "rb") as f:
            files = {'image': (file_path.name, f, 'image/jpeg')}
            response = httpx.post(API_URL, files=files, timeout=10.0)
            
        if response.status_code == 200:
            result = response.json()
            nav = result["navigation_command"]
            
            # Create command for FalconSim
            command = {
                "speed": nav["speed_kmh"],
                "heading": nav["heading"],
                "stop": nav["stop"],
                "timestamp": datetime.now().isoformat()
            }
            
            # Write output file
            output_filename = file_path.stem + ".json"
            output_path = FALCONSIM_OUTPUT_DIR / output_filename
            
            with open(output_path, "w") as f:
                json.dump(command, f, indent=2)
                
            print(f" ✅ Command written to: {output_path.name}")
            
            # Clean up input frame (optional, usually simulation handles this or we move it)
            # os.remove(file_path) 
            return True
        else:
            print(f" ❌ Server error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f" ❌ Error processing frame: {e}")
        return False

def main():
    print("================================")
    print("   DUALITY FALCONSIM BRIDGE")
    print("================================")
    
    ensure_dirs()
    
    print(f"FalconSim bridge active")
    print(f"Watching: {FALCONSIM_INPUT_DIR}")
    print(f"Commands → {FALCONSIM_OUTPUT_DIR}")
    print("Ready for rover connection. Press Ctrl+C to stop.")
    
    processed_files = set()
    
    # Pre-populate processed files to avoid re-processing old data on start
    for f in FALCONSIM_INPUT_DIR.glob("*.jpg"):
        processed_files.add(f.name)
        
    try:
        while True:
            # Look for new JPG files
            current_files = list(FALCONSIM_INPUT_DIR.glob("*.jpg"))
            
            for file_path in current_files:
                if file_path.name not in processed_files:
                    success = process_frame(file_path)
                    if success:
                        processed_files.add(file_path.name)
            
            time.sleep(POLL_INTERVAL_MS / 1000.0)
            
    except KeyboardInterrupt:
        print("\nBridge stopped.")

if __name__ == "__main__":
    main()
