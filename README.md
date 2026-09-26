# WheelBuddy

**WheelBuddy** is an AI-powered iOS driving companion designed to bring modern driver-assistance features to vehicles using only an iPhone.

The app combines **real-time driver monitoring, drowsiness detection, road perception, lane assistance, pedestrian and collision-risk analysis, safety-aware route selection, speed monitoring, and turn-by-turn navigation** in one system.

Built for **ShellHacks 2026**.

---

## Overview

Many modern vehicles include advanced driver-assistance systems such as driver monitoring, lane assistance, collision warnings, speed-limit alerts, and intelligent navigation.

WheelBuddy explores how many of these capabilities can be provided using the sensors and compute already available on a smartphone.

The system combines:

- On-device computer vision
- Core ML inference
- Apple Vision facial analysis
- Front and rear camera processing
- GPS and motion data
- Google Maps and Navigation
- Machine-learning-based crime prediction
- Temporal drowsiness detection
- FastAPI-based AI inference services

The goal is to create a single mobile driving companion that can continuously monitor the **driver, road, route, and surrounding environment**.

---

# Features

## Driver Monitoring

WheelBuddy uses the iPhone's **front-facing camera** to monitor the driver.

Current capabilities include:

- Face detection
- Head pose estimation
- Driver-attention monitoring
- Looking-away detection
- Eye feature extraction
- Driver presence detection
- Drowsiness feature collection

Apple's **Vision framework** is used to extract facial information in real time.

---

## Drowsiness Detection

WheelBuddy includes a machine-learning-based drowsiness detection pipeline.

Facial and eye features are extracted on the iPhone and sent to a local inference API running a **causal Temporal Convolutional Network (TCN)**.

The current feature schema contains 12 temporal features:

```text
face_detected
yaw
pitch
roll
left_eye_valid
right_eye_valid
left_eye_aspect_ratio
right_eye_aspect_ratio
left_pupil_rel_x
left_pupil_rel_y
right_pupil_rel_x
right_pupil_rel_y
```

The model predicts three driver-eye states:

```text
closed
open
undefined
```

The temporal model allows WheelBuddy to analyze driver behavior across multiple frames rather than relying on a single image.

---

## 📷 Dual-Camera ADAS

WheelBuddy uses **AVCaptureMultiCamSession** to process both cameras simultaneously:

- **Front camera:** driver monitoring and drowsiness features
- **Rear camera:** road perception and lane detection

This enables simultaneous monitoring of the driver and the road.

---

## Road Object Detection

The rear camera is processed using an on-device **Core ML object detector**.

The current road perception system can detect road users such as:

- Cars
- Trucks
- Buses
- Motorcycles
- Bicycles
- Pedestrians

The detections are processed by additional risk-analysis modules instead of being used only as visual bounding boxes.

---

## Forward Road-Risk Detection

WheelBuddy analyzes detected vehicles over time to identify potential forward hazards.

The risk pipeline considers signals such as:

- Object position
- Distance from the center of the driving corridor
- Bounding-box scale
- Scale expansion over time
- Object tracking
- Relative motion

The system can transition through different warning states as risk increases.

---

## 🚶 Pedestrian Risk Detection

Detected pedestrians are analyzed separately to determine whether they may present an immediate driving hazard.

The pedestrian risk system considers factors including:

- Pedestrian position
- Road-relative location
- Tracking history
- Motion
- Proximity to the vehicle's projected path

---

## Lane Assistance

WheelBuddy includes lane detection using an on-device **Core ML lane-detection model**.

The current model package is based on:

```text
UFLDv2_CurveLanes_ResNet18
```

Lane geometry is converted into lane-assist states that can be displayed in the driving interface and used by the alert system.

WheelBuddy currently includes both:

- Core ML lane-detection backend
- Geometric lane-processing backend

---

## Navigation

WheelBuddy integrates Google's iOS mapping ecosystem.

Navigation features include:

- Live Google Map
- Current location
- Destination search
- Places autocomplete
- Multiple route alternatives
- Route selection
- Route polylines
- ETA
- Distance
- Turn-by-turn navigation
- Re-centering
- Navigation-mode UI

---

## Safety-Aware Routing

WheelBuddy does not evaluate routes using travel time alone.

Candidate routes can also be analyzed using a custom **crime-rate prediction model**.

For each route, WheelBuddy evaluates the geographic areas and time periods through which the driver will travel.

The app can then present alternative routes with safety information alongside traditional navigation information such as:

- ETA
- Distance
- Route geometry

---

## CrimePredictor

The crime prediction system is located in:

```text
AIPackages/CrimePredictor
```

It uses a PyTorch neural network to estimate **crime rates** for geographic areas and time periods.

### Input features

The current crime model uses features including:

```text
H3 cell
city
month
day of week
time bin
weekend indicator
```

Geographic regions are represented using **H3 spatial indexing**.

The current system uses:

```text
H3 resolution: 9
```

The model predicts per-category crime rates rather than simple binary crime/no-crime classifications.

These predictions are then aggregated along candidate navigation routes.

---

## Time-Dependent Route Safety

Route safety can vary depending on the time of day.

WheelBuddy therefore analyzes route traversal timing rather than assigning a permanently fixed safety score to a location.

Crime predictions are calculated using temporal bins and can be used to display how route safety changes depending on departure time.

---

## Speed Monitoring

WheelBuddy monitors the vehicle's current speed using GPS.

The system includes support for:

- Current vehicle speed
- Posted speed limit
- Overspeed detection
- Speed-related warnings

Posted speed limits can be retrieved using Google's road services.

---

## ADAS Alerts

WheelBuddy includes a centralized ADAS alert system.

Warnings can use:

- Visual alerts
- Audio alerts
- Haptic feedback

The alert manager coordinates warnings generated by different subsystems to avoid inconsistent behavior.

Examples include:

- Driver not detected
- Driver looking away
- Drowsiness
- Forward collision risk
- Pedestrian risk
- Lane departure
- Overspeeding

---

# Tech Stack

## iOS

| Technology | Purpose |
| --- | --- |
| **Swift** | Main application language |
| **SwiftUI** | User interface |
| **AVFoundation** | Camera capture and MultiCam |
| **Vision** | Face and facial-landmark analysis |
| **Core ML** | On-device AI inference |
| **Core Location** | GPS and vehicle speed |
| **Google Maps SDK for iOS** | Map rendering |
| **Google Places SDK** | Destination search |
| **Google Navigation SDK** | Turn-by-turn navigation |
| **Google Routes API** | Route generation |
| **Google Roads API** | Road and speed-limit information |
| **Swift Package Manager** | iOS dependency management |

---

## AI / Backend

| Technology | Purpose |
| --- | --- |
| **Python** | AI training and inference |
| **PyTorch** | Neural-network models |
| **FastAPI** | AI inference APIs |
| **Uvicorn** | ASGI application server |
| **H3** | Geographic spatial indexing |
| **NumPy** | Numerical processing |
| **Pandas** | Dataset processing |
| **PyArrow** | Parquet data processing |
| **DuckDB** | Large dataset querying |
| **Pydantic** | API schemas and validation |
| **Pytest** | Python testing |

---

# System Architecture

```text
                         ┌─────────────────────────┐
                         │       Google APIs       │
                         │                         │
                         │ Maps                    │
                         │ Places                  │
                         │ Routes                  │
                         │ Navigation              │
                         │ Roads                   │
                         └────────────┬────────────┘
                                      │
                                      ▼
┌───────────────────┐       ┌─────────────────────────────┐
│   Front Camera    │──────▶│                             │
│                   │       │                             │
│ Driver Monitoring │       │         WheelBuddy          │
│ Eye Features      │       │          iOS App            │
└───────────────────┘       │                             │
                            │ Swift / SwiftUI              │
┌───────────────────┐       │ AVFoundation                │
│    Rear Camera    │──────▶│ Vision / Core ML            │
│                   │       │ Core Location               │
│ Road Detection    │       │                             │
│ Lane Detection    │       └──────────┬──────────┬───────┘
└───────────────────┘                  │          │
                                       │          │ HTTP
                           On-device   │          │
                           inference   ▼          ▼
                                ┌──────────┐ ┌─────────────────────┐
                                │ Core ML  │ │ Python AI Services  │
                                │          │ │                     │
                                │ Road     │ │ CrimePredictor      │
                                │ Objects  │ │ Port 8000           │
                                │ Lanes    │ │                     │
                                └──────────┘ │ DrowsinessDetection │
                                             │ Port 8001           │
                                             └─────────────────────┘
```

---

# Repository Structure

```text
WheelBuddy/
│
├── WheelBuddy/
│   │
│   ├── Models/
│   │   ├── DriverState.swift
│   │   ├── LaneAssistState.swift
│   │   ├── NavigationMode.swift
│   │   ├── NavigationSessionModel.swift
│   │   ├── PedestrianRisk.swift
│   │   ├── RoadDetection.swift
│   │   ├── RoadRisk.swift
│   │   └── SpeedingState.swift
│   │
│   ├── Services/
│   │   │
│   │   ├── Alerts/
│   │   │   └── ADASAlertManager.swift
│   │   │
│   │   ├── Camera/
│   │   │   ├── CameraManager.swift
│   │   │   └── MultiCamManager.swift
│   │   │
│   │   ├── Crime/
│   │   │   ├── CrimeAPIConfiguration.swift
│   │   │   ├── CrimePredictionService.swift
│   │   │   └── RoutePredictionModels.swift
│   │   │
│   │   ├── Drowsiness/
│   │   │   ├── DrowsinessAPIClient.swift
│   │   │   ├── DrowsinessAPIConfiguration.swift
│   │   │   ├── DrowsinessFeatureExtractor.swift
│   │   │   └── DrowsinessInferenceCoordinator.swift
│   │   │
│   │   ├── Location/
│   │   │   ├── RoadsSpeedLimitService.swift
│   │   │   └── SpeedMonitor.swift
│   │   │
│   │   ├── Places/
│   │   │   ├── PlacesAutocompleteService.swift
│   │   │   └── UserLocationProvider.swift
│   │   │
│   │   ├── Risk/
│   │   │   ├── PedestrianRiskAnalyzer.swift
│   │   │   └── RoadRiskAnalyzer.swift
│   │   │
│   │   ├── Routes/
│   │   │   ├── GoogleRoutesService.swift
│   │   │   ├── RouteExtractionPayload.swift
│   │   │   └── RouteRiskScorer.swift
│   │   │
│   │   └── Vision/
│   │       ├── CoreMLLaneBackend.swift
│   │       ├── DriverMonitor.swift
│   │       ├── GeometricLaneBackend.swift
│   │       ├── LaneDetectionService.swift
│   │       └── RoadDetectionService.swift
│   │
│   ├── Views/
│   │   └── SwiftUI application views
│   │
│   └── ML/
│       ├── RoadObjectDetector.mlpackage
│       └── UFLDv2_CurveLanes_ResNet18.mlpackage
│
├── WheelBuddyTests/
│   └── iOS tests
│
├── AIPackages/
│   │
│   ├── CrimePredictor/
│   │   ├── api/
│   │   ├── data/
│   │   ├── scripts/
│   │   ├── src/
│   │   └── tests/
│   │
│   └── DrowsinessDetection/
│       ├── api/
│       ├── data/
│       ├── model/
│       ├── scripts/
│       └── tests/
│
├── Secrets.example.xcconfig
├── WheelBuddy.xcconfig
├── InfoPlistAdditions.plist
├── start_servers.txt
├── LICENSE
└── README.md
```

---

# Getting Started

## Requirements

### iOS

You will need:

- macOS
- Xcode
- iPhone for full hardware testing
- Apple Developer signing configuration
- Google Cloud project
- Required Google API keys

A physical iPhone is strongly recommended because several WheelBuddy features depend on:

- Camera
- MultiCam
- GPS
- Motion sensors
- Navigation
- Local-network communication

---

# Clone the Repository

```bash
git clone https://github.com/mazid-rafee/WheelBuddy.git
cd WheelBuddy
```

Open the Xcode project:

```bash
open WheelBuddy.xcodeproj
```

---

# Google API Configuration

WheelBuddy uses Google services including:

- Maps SDK for iOS
- Places API / Places SDK
- Navigation SDK for iOS
- Routes API
- Roads API

Create your local secrets configuration:

```bash
cp Secrets.example.xcconfig Secrets.xcconfig
```

Then edit:

```text
Secrets.xcconfig
```

and provide your API keys:

```text
MAPS_API_KEY = YOUR_MAPS_API_KEY
ROUTES_API_KEY = YOUR_ROUTES_API_KEY

DROWSINESS_API_KEY = YOUR_OPTIONAL_DROWSINESS_API_KEY
```

> **Never commit `Secrets.xcconfig` to GitHub.**

API keys should also be restricted through Google Cloud.

---

# Python AI Services

WheelBuddy currently runs two development inference servers.

```text
CrimePredictor
Port: 8000

DrowsinessDetection
Port: 8001
```

---

# Running CrimePredictor

Navigate to:

```bash
cd AIPackages/CrimePredictor
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Start the server:

```bash
python -m uvicorn api.app:app \
    --host 0.0.0.0 \
    --port 8000
```

Test the health endpoint:

```bash
curl http://127.0.0.1:8000/health
```

---

# Running DrowsinessDetection

Open another terminal:

```bash
cd AIPackages/DrowsinessDetection
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r api/requirements.txt
```

Run the inference server:

```bash
python -m uvicorn api.app:app \
    --host 0.0.0.0 \
    --port 8001
```

The drowsiness API can optionally require:

```text
X-API-Key
```

through the `DROWSINESS_API_KEY` environment/configuration value.

---

# Using a Physical iPhone

When using an iPhone, `127.0.0.1` points to the phone itself, not the development Mac.

The Mac and iPhone should therefore be connected to the same network.

Find the Mac's Wi-Fi IP:

```bash
ipconfig getifaddr en0
```

Example:

```text
192.168.1.25
```

Update the device URLs in:

```text
WheelBuddy.xcconfig
```

Example:

```text
CRIME_API_BASE_URL[sdk=iphoneos*] = http:/$()/192.168.1.25:8000

DROWSINESS_API_BASE_URL[sdk=iphoneos*] = http:/$()/192.168.1.25:8001
```

Simulator builds can continue using:

```text
127.0.0.1
```

---

# Core ML Models

WheelBuddy includes Core ML models directly in the iOS project.

## Road Object Detector

```text
WheelBuddy/ML/RoadObjectDetector.mlpackage
```

Used for road-user detection.

---

## Lane Detection

```text
WheelBuddy/ML/UFLDv2_CurveLanes_ResNet18.mlpackage
```

Used by the Core ML lane-detection backend.

Large model weights may be managed using **Git LFS**.

---

# Required Permissions

WheelBuddy requires several iOS permissions.

Depending on the enabled features these include:

- Camera access
- Location access
- Background location
- Motion access
- Local-network access

Local-network access is primarily required during development when the iPhone communicates with AI services running on a computer.

---

# Testing

## iOS Tests

WheelBuddy includes:

```text
WheelBuddyTests
```

Run tests directly from Xcode.

---

## CrimePredictor Tests

```bash
cd AIPackages/CrimePredictor
pytest
```

---

## DrowsinessDetection Tests

```bash
cd AIPackages/DrowsinessDetection
pytest
```

The drowsiness package includes tests for areas such as:

- API behavior
- Feature schemas
- Data loading
- Feature extraction
- Inference
- Eye-state processing

---

# Security

When developing or deploying WheelBuddy:

- Never commit API keys.
- Never commit private credentials.
- Keep `Secrets.xcconfig` local.
- Restrict Google API keys.
- Do not expose local FastAPI servers directly to the public internet.
- Use authentication and TLS for production deployments.
- Store server credentials securely.
- Avoid placing sensitive user information in logs.

---

# Privacy

WheelBuddy processes potentially sensitive information including:

- Camera frames
- Facial landmarks
- Eye features
- Vehicle location
- Route information

Whenever possible, latency-sensitive computer-vision processing is performed directly on the device.

Any production deployment should clearly disclose:

- What information is collected
- What information leaves the device
- How information is stored
- How long information is retained
- Whether information is shared with external services

---

# Safety Notice

> **WheelBuddy is an experimental driver-assistance application and is not a replacement for attentive driving or certified vehicle safety systems.**

Computer-vision detections, AI predictions, route safety estimates, navigation information, speed limits, and alerts may be:

- Incorrect
- Delayed
- Incomplete
- Unavailable

Drivers remain responsible for:

- Paying attention to the road
- Controlling the vehicle
- Following traffic laws
- Following official road signs
- Making safe driving decisions

Do not rely on WheelBuddy as the sole source of safety-critical driving information.

---

# Current Development Areas

WheelBuddy is actively being developed in areas including:

- Driver attention monitoring
- Temporal drowsiness detection
- Road-risk estimation
- Pedestrian-risk detection
- Lane detection
- Lane departure assistance
- Safety-aware route ranking
- Time-dependent route safety
- Crime-rate prediction
- Vehicle speed monitoring
- Speed-limit monitoring
- ADAS alerts
- Navigation UI
- Real-device performance optimization

---

# Future Improvements

Potential future improvements include:

- Fully on-device drowsiness inference
- Improved lane geometry estimation
- More robust nighttime perception
- Better forward-collision estimation
- Traffic-aware safety scoring
- Weather-aware risk estimation
- Improved pedestrian trajectory prediction
- Production cloud inference infrastructure
- Model quantization and acceleration
- Apple CarPlay integration
- Expanded geographic coverage for route-risk prediction

---

# Contributing

Contributions, bug reports, and suggestions are welcome.

A typical contribution workflow is:

```bash
git checkout -b feature/my-feature
```

Make your changes and commit:

```bash
git add .
git commit -m "Add my feature"
```

Push the branch:

```bash
git push -u origin feature/my-feature
```

Then open a pull request.

For major features, please describe:

- What was changed
- Why it was changed
- How it was tested
- Any new dependencies
- Any changes to configuration

---

# License

WheelBuddy is distributed under the **MIT License**.

See:

```text
LICENSE
```

for the complete license.

---

# Authors

**Md Abdullah Al Mazid**

GitHub: [@mazid-rafee](https://github.com/mazid-rafee)

**Zaima Zarnaz**

GitHub: [@zaima14zarnaz](https://github.com/zaima14zarnaz)

Repository:

[github.com/mazid-rafee/WheelBuddy](https://github.com/mazid-rafee/WheelBuddy)

---

## ShellHacks 2026 🐚

WheelBuddy was developed as a **ShellHacks 2026** project with the goal of exploring how smartphone sensors, computer vision, machine learning, and modern navigation services can be combined to provide an accessible intelligent driving-assistance platform.

---

### Drive smarter. Stay aware. Stay safe.