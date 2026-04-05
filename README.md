# 🦾 Scapula Push-Up Tracker

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-orange.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A real-time, markerless scapulothoracic joint tracker for evaluating push-up form. Built using **Python**, **OpenCV**, and **Google's MediaPipe Pose**, this tool evaluates Scapulohumeral Rhythm (SHR) and scapular kinematics against established clinical biomechanics literature.

No model training or expensive depth-cameras required—works completely locally on a standard webcam.

<p align="center">
  <img src="demo_screenshot.png" alt="Scapula Tracker HUD Demo" width="700">
</p>

## ✨ Key Features
* **Markerless Pose Estimation:** Uses a 2D proxy for scapular upward rotation via MediaPipe's lightweight topology.
* **Live Biomechanical Analysis:** Computes Scapulohumeral Rhythm (SHR) dynamically as the ratio of glenohumeral elevation to scapular upward rotation ($\Delta$GH / $\Delta$scap).
* **Rule-Based Form Classifier:** Detects insufficient rotation, excessive elevation, impingement risks, and left/right winging asymmetries.
* **Real-Time HUD:** Features rolling sparkline graphs for kinematic tracking, rep counting, and a dynamic 0-100 quality score.
* **Session Logging:** Automatically exports detailed kinematics for every rep to a CSV file for post-session analysis.
* **Hybrid Tracking (Optional):** Built-in HSV color-masking to track physical adhesive dots (e.g., on the inferior angle and acromion) for enhanced precision.

## 🔬 Clinical Validity & Literature Basis
The correctness thresholds and tracking logic are directly grounded in peer-reviewed orthopedic and biomechanical literature:
* **Ludewig & Cook (2000)** *JOSPT* — Baseline norms for scapular motion during a push-up.
* **McClure et al. (2001)** *JSES* — Establishes 30–45° upward rotation at the top of the concentric phase.
* **Ludewig et al. (2004)** *AJSM* — Serratus anterior balance and winging implications.
* **Inman (1944) / Bagg & Forrest (1988)** — Defines the healthy 2:1 Scapulohumeral Rhythm (SHR).

## 🚀 Quick Start

### 1. Install Dependencies
Clone this repository and install the required packages:
```bash
git clone [https://github.com/YOUR_USERNAME/scapula-motion-tracker.git](https://github.com/YOUR_USERNAME/scapula-motion-tracker.git)
cd scapula-motion-tracker
pip install -r requirements.txt