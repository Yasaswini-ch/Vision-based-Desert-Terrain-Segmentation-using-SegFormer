import os
import io
import time
import base64
import json
import asyncio
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import websockets

import torchvision.transforms as T
import albumentations as A
import sqlite3

import config
import model
from dataset import get_val_test_transforms

app = Flask(__name__)
CORS(app)

# Global state
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seg_model = None
transform = None
last_nav_command = {
    "speed_kmh": 0.0,
    "heading": "stop",
    "corridor": "center",
    "stop": True,
    "confidence": 0.0
}
frames_processed = 0
start_time = time.time()

# Monitoring clients
monitor_clients = set()

def load_inference_model():
    global seg_model, transform
    print(f"Loading model on {device}...")
    
    # Load model architecture
    seg_model = model.load_model(
        num_classes=config.NUM_CLASSES,
        pretrained=False,
        backend=config.MODEL_BACKEND,
        model_name=config.HF_MODEL_NAME
    )
    
    # Load weights
    if config.BEST_MODEL_PATH.exists():
        checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=device, weights_only=False)
        if "model_state_dict" in checkpoint:
            seg_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            seg_model.load_state_dict(checkpoint)
        print(f"Loaded weights from {config.BEST_MODEL_PATH}")
    else:
        print("Warning: Best model checkpoint not found. Using uninitialized weights.")
    
    seg_model.to(device)
    seg_model.eval()
    
    # Setup transform
    transform = get_val_test_transforms(config.IMAGE_SIZE)

def get_navigation_logic(mask_np):
    """
    Divide image into 3 columns (left/center/right)
    Find safest corridor (most traversable pixels: classes 2 and 8)
    """
    h, w = mask_np.shape
    col_w = w // 3
    
    # Traversable classes: Dry Grass (2), Landscape (8)
    traversable_mask = np.logical_or(mask_np == 2, mask_np == 8)
    # Obstacle classes: Trees (0), Lush Bushes (1), Logs (6), Rocks (7)
    obstacle_mask = np.isin(mask_np, [0, 1, 6, 7])
    
    corridors = {
        "left": traversable_mask[:, :col_w],
        "center": traversable_mask[:, col_w:2*col_w],
        "right": traversable_mask[:, 2*col_w:]
    }
    
    obstacle_corridors = {
        "left": obstacle_mask[:, :col_w],
        "center": obstacle_mask[:, col_w:2*col_w],
        "right": obstacle_mask[:, 2*col_w:]
    }
    
    corridor_scores = {k: np.mean(v) for k, v in corridors.items()}
    obstacle_scores = {k: np.mean(v) for k, v in obstacle_corridors.items()}
    
    total_traversable = np.mean(traversable_mask) * 100
    total_obstacles = np.mean(obstacle_mask) * 100
    
    # Heading logic
    safest_corridor = max(corridor_scores, key=corridor_scores.get)
    
    heading = "forward"
    if safest_corridor == "left":
        heading = "bear_left"
    elif safest_corridor == "right":
        heading = "bear_right"
        
    # Stop logic
    stop = False
    if total_obstacles > 80:
        heading = "emergency_stop"
        stop = True
    elif total_obstacles > 60:
        heading = "stop"
        stop = True
        
    # Speed logic
    speed = 0.0
    if not stop:
        if total_traversable > 70: speed = 8.0
        elif total_traversable > 50: speed = 5.0
        elif total_traversable > 30: speed = 2.5
        elif total_traversable > 10: speed = 1.0
        else:
            speed = 0.0
            heading = "stop"
            stop = True
            
    # Hazards
    hazards = []
    if np.mean(mask_np == 7) > 0.03: hazards.append("rocks_detected")
    if np.mean(mask_np == 6) > 0.01: hazards.append("logs_present")
    
    risk_level = "LOW"
    if total_obstacles > 40: risk_level = "CRITICAL"
    elif total_obstacles > 25: risk_level = "HIGH"
    elif total_obstacles > 10: risk_level = "MEDIUM"
    
    return {
        "speed_kmh": speed,
        "heading": heading,
        "corridor": safest_corridor,
        "stop": stop,
        "confidence": 0.87, # Placeholder for mean model confidence
        "hazards": hazards,
        "risk_level": risk_level,
        "traversable_pct": float(total_traversable)
    }

def run_inference(image_bytes):
    global frames_processed, last_nav_command
    start_inf = time.time()
    
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_w, orig_h = img.size
    
    # Preprocess
    img_np = np.array(img)
    augmented = transform(image=img_np)
    input_tensor = augmented["image"].unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        logits = model.forward_logits(seg_model, input_tensor, backend=config.MODEL_BACKEND)
        # Upsample if needed
        if logits.shape[-2:] != (config.IMAGE_SIZE, config.IMAGE_SIZE):
            logits = F.interpolate(logits, size=(config.IMAGE_SIZE, config.IMAGE_SIZE), mode="bilinear", align_corners=False)
        
        probs = F.softmax(logits, dim=1)
        mean_confidence = float(probs.max(dim=1)[0].mean().cpu().item())
        mask = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy()
        
    inf_time_ms = int((time.time() - start_inf) * 1000)
    frames_processed += 1
    
    # Class distribution
    unique, counts = np.unique(mask, return_counts=True)
    dist = {config.CLASS_NAMES[int(u)]: float(c / mask.size * 100) for u, c in zip(unique, counts)}
    
    # Navigation logic
    nav = get_navigation_logic(mask)
    nav["confidence"] = mean_confidence
    last_nav_command = nav
    
    # Mask overlay (Base64)
    # Create colored mask
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for cls_idx, color in config.CLASS_COLORS.items():
        color_mask[mask == cls_idx] = color
        
    mask_img = Image.fromarray(color_mask)
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    mask_b64 = base64.b64encode(buf.getvalue()).decode()
    response = {
        "segmentation_mask": mask_b64,
        "class_distribution": dist,
        "navigation_command": nav,
        "hazards": nav["hazards"],
        "risk_level": nav["risk_level"],
        "traversable_pct": nav["traversable_pct"],
        "inference_time_ms": inf_time_ms,
        "timestamp": datetime.now().isoformat()
    }
    
    # Log to SQLite DB
    try:
        db_path = config.PROJECT_ROOT / "runs" / "logs" / "mission_history.db"
        if db_path.exists():
            conn = sqlite3.connect(str(db_path))
            c = conn.cursor()
            dom_class = max(dist.items(), key=lambda x: x[1])[0] if dist else "Unknown"
            c.execute('''
                INSERT INTO segmentation_runs 
                (timestamp, image_filename, mean_iou, risk_level, traversable_pct, obstacle_pct, dominant_class, weather_location, processing_time_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                response["timestamp"], 
                "Live_Stream_Frame", 
                mean_confidence, 
                nav["risk_level"], 
                nav["traversable_pct"], 
                100 - nav["traversable_pct"], 
                dom_class, 
                "Unknown", 
                inf_time_ms
            ))
            conn.commit()
            conn.close()
    except Exception as e:
        print(f"DB Log Error: {e}")

    # Broadcast to monitor clients
    if monitor_clients:
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(broadcast_to_monitors(response))
        except RuntimeError:
            if 'ws_loop' in globals() and ws_loop is not None:
                asyncio.run_coroutine_threadsafe(broadcast_to_monitors(response), ws_loop)
        
    return response

async def broadcast_to_monitors(data):
    if not monitor_clients:
        return
    message = json.dumps(data)
    disconnected = set()
    for ws in monitor_clients:
        try:
            await ws.send(message)
        except:
            disconnected.add(ws)
    for ws in disconnected:
        monitor_clients.remove(ws)

# Flask Routes
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "operational",
        "model": "segformer_b2",
        "mean_iou": 0.6442,
        "device": str(device),
        "uptime_seconds": int(time.time() - start_time),
        "frames_processed": frames_processed
    })

@app.route('/segment', methods=['POST'])
def segment():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    file = request.files['image']
    image_bytes = file.read()
    response = run_inference(image_bytes)
    return jsonify(response)

@app.route('/segment/batch', methods=['POST'])
def segment_batch():
    if 'images' not in request.files:
        files = request.files.getlist('images')
    else:
        files = request.files.getlist('images')
        
    if not files:
        return jsonify({"error": "No image files provided"}), 400
        
    responses = []
    for file in files:
        image_bytes = file.read()
        responses.append(run_inference(image_bytes))
    return jsonify(responses)

@app.route('/navigation/current', methods=['GET'])
def current_navigation():
    return jsonify(last_nav_command)

import math
import requests

def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)

@app.route('/api/location', methods=['GET'])
def api_location():
    location_name = request.args.get('q', '')
    if not location_name:
        err = jsonify({"error": "Empty query"})
        err.headers.add("Access-Control-Allow-Origin", "*")
        return err, 400
    try:
        geocode_url = f"https://nominatim.openstreetmap.org/search?q={requests.utils.quote(location_name)}&format=json&limit=1"
        headers = {'User-Agent': 'DesertSegStudio/2.0'}
        geo_req = requests.get(geocode_url, headers=headers, timeout=5)
        geo_data = geo_req.json()
        if not geo_data:
            err = jsonify({"error": "Location not found"})
            err.headers.add("Access-Control-Allow-Origin", "*")
            return err, 404
            
        lat = float(geo_data[0]["lat"])
        lon = float(geo_data[0]["lon"])
        display_name = geo_data[0]["display_name"]
        osm_type = geo_data[0].get("osm_type", "unknown")
            
        elev_req = requests.get(f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}", timeout=5)
        elev_data = elev_req.json()
        elevation = elev_data["results"][0]["elevation"] if "results" in elev_data else 0
        
        xt, yt = deg2num(lat, lon, 12)
        
        res = jsonify({
            "data": {
                "lat": lat, "lon": lon, "name": display_name,
                "elevation": elevation, "type": osm_type,
                "xtile": xt, "ytile": yt
            }
        })
        res.headers.add("Access-Control-Allow-Origin", "*")
        return res
    except Exception as e:
        err = jsonify({"error": str(e)})
        err.headers.add("Access-Control-Allow-Origin", "*")
        return err, 500

@app.route('/stream/mjpeg')
def mjpeg_stream():
    def generate():
        while True:
            # This is a bit tricky without a continuous source here, 
            # but we can return the last processed frame's mask if needed
            # For a real stream, we'd need a way to trigger this.
            # MJPEG stream implementation usually requires a source.
            # We'll return a placeholder or wait for a frame.
            time.sleep(0.2)
            # Placeholder or latest frame logic here if implemented
            pass
    return "MJPEG Stream endpoint ready (Requires active frame source)", 200

# WebSocket Server
async def rover_handler(websocket):
    print("Rover connected")
    try:
        async for message in websocket:
            # message is raw JPEG binary
            response = run_inference(message)
            await websocket.send(json.dumps(response["navigation_command"]))
    except websockets.exceptions.ConnectionClosed:
        print("Rover disconnected")

async def monitor_handler(websocket):
    print("Monitor client connected")
    monitor_clients.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        monitor_clients.remove(websocket)

async def start_ws():
    print("WebSocket servers starting on ports 8001 (rover) and 8002 (monitor)...")
    async with websockets.serve(rover_handler, "0.0.0.0", 8001, max_size=10 * 1024 * 1024), \
               websockets.serve(monitor_handler, "0.0.0.0", 8002, max_size=10 * 1024 * 1024):
        await asyncio.Future() # run forever

ws_loop = None

def start_websocket_server():
    global ws_loop
    loop = asyncio.new_event_loop()
    ws_loop = loop
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(start_ws())
    except Exception as e:
        print(f"WebSocket server error: {e}")

if __name__ == "__main__":
    load_inference_model()
    
    # Start WebSockets in a separate thread
    ws_thread = threading.Thread(target=start_websocket_server, daemon=True)
    ws_thread.start()
    
    # Start Flask
    print("Inference Server (Flask) starting on http://0.0.0.0:5001...")
    app.run(host='0.0.0.0', port=5001, threaded=True)
