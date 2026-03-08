# 🏜️ Desert Rover Intelligence System: Hackathon Demo Guide

This document provides a step-by-step walkthrough for demonstrating the **Desert Rover Intelligence System** to judges.

---

## 🏗️ System Overview
Our system is a **Live Rover Intelligence System** that transforms static segmentation into autonomous action. It consists of:
1.  **The Brain (`inference_server.py`)**: A Flask-based inference engine running SegFormer B2.
2.  **The Eyes (`app.py`)**: A real-time monitoring dashboard for mission control.
3.  **The Rover (`rover_simulator.py`)**: A digital twin simulating field operations.

---

## 🚀 Step 1: Rapid Launch
1.  Double-click `start_rover_system.bat`.
2.  Wait for the **Inference Server** and **Monitor Dashboard** to initialize.
3.  When prompted in the terminal, press `S` to launch the **Rover Simulator**.

---

## 🤖 Step 2: The Live Monitor (The "Wow" Factor)
1.  Open the Streamlit dashboard in your browser.
2.  Click the **"🤖 Live Rover Feed"** tab.
3.  **What to show the judges:**
    *   **Live Overlay**: Show the real-time segmentation mask being sent over WebSockets.
    *   **Navigation Status**: Point out the dynamic command (e.g., "BEAR LEFT", "FORWARD") changing based on the terrain.
    *   **Risk Level**: Show how the system flags high-risk areas (e.g., "HIGH RISK" when obstacles dominate).
    *   **Mission Timeline**: Demonstrate the live speed and traversability charts tracking mission health.

---

## 📂 Step 3: Simulation Integration (Technical Deep Dive)
1.  **FalconSim Bridge**: Explain that `falconsim_bridge.py` allows the system to integrate with Duality's FalconSim by polling for exported frames and writing motor commands to a JSON file.
2.  **ROS2 Ready**: Mention the `ros2_bridge.py` which provides out-of-the-box support for ROS2 Humble/Foxy via standard `Twist` messages.

---

## 📊 Step 4: Analytical Deep Dive
1.  Switch to the **"Failure Intelligence"** tab.
2.  Show how the system automatically identifies the "Hardest Images" where the model is most uncertain.
3.  Explain that this creates a virtuous cycle of **Active Learning** for future model iterations.

---

## 🏆 Summary for Judges
> "We didn't just build a model; we built a complete autonomous stack. From real-time inference to mission telemetry, our system is ready to be deployed on a physical rover today."
