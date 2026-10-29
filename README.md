# FarmHawk: Autonomous Precision Drone Scouting & Threat Intelligence Ecosystem

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![React 18](https://img.shields.io/badge/react-18.3-61DAFB.svg)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF3838.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

FarmHawk is an end-to-end autonomous precision agricultural drone ecosystem engineered to protect crops, slash agrochemical overuse, and safeguard rural farm livelihoods. 

By unifying high-resolution aerial computer vision, deep convolutional neural networks (**YOLOv8**), microclimate agronomic risk modeling, automated rural SMS gateways (**TextBee SIM relay**), and a collaborative farmer knowledge network (**Krishi Samvad**), FarmHawk empowers smallholder and commercial farmers with sub-second foliar disease and pest diagnostics.

---

## System Architecture

FarmHawk operates across four resilient, decoupled layers:

![FarmHawk Architecture](farmhawk_architecture.jpg)

1. **Aerial Acquisition & Edge Computer Vision**: High-resolution drone video streams processed through a dual-stream YOLOv8 model capable of diagnosing 35 agricultural classes (29 foliar diseases and 6 insect pests) with distinct color-coded telemetry.
2. **Micro-Climate Agronomic Intelligence Engine**: Real-time ambient temperature, humidity, precipitation, and wind tracking correlated against crop phenology (e.g. Day 54 Heading & Flowering) to forecast fungal sporulation risk before visual lesions erupt.
3. **24/7 Autonomous Climate Sentinel & SMS Dispatcher**: Background monitoring service that detects imminent weather shocks (Heavy Rain, Heatwaves, Frost) and instantly dispatches bilingual (English / Hindi) emergency instructions to registered farmers via SIM-connected mobile gateways.
4. **Krishi Samvad & Community Intelligence**: Peer-to-peer knowledge sharing network connecting rural farmers with agronomic experts, real-time threat maps, and localized treatment recommendations.

---

## Empirical Economic Impact & Feasibility

Field trials and structured surveys across 20 commercial agricultural operations demonstrated significant sustainability and yield gains:

| Economic & Ecological Metric | FarmHawk Impact |
| :--- | :--- |
| **Chemical Fungicide Usage** | **-30% to -45%** reduction via precision localized spraying |
| **Agricultural Water Consumption** | **-90%** water savings in drone ultra-low-volume misting |
| **Field Scouting Labor Hours** | **-85%** time reduction compared to manual perimeter walks |
| **Harvestable Yield Recovery** | **+12% to +22%** saved from pre-symptomatic blight containment |

![Economic Impact](farmhawk_economic_impact.png)
![Feasibility Radar](farmhawk_feasibility_radar.png)

---

## Repository Structure

```
Farmhawk/
├── backend/                        # FastAPI REST API & Deep Learning Subsystem
│   ├── models/                     # YOLOv8 PyTorch model weights (.pt)
│   ├── routes/                     # Modular API endpoints (Fields, Notifications, Community, Survey)
│   ├── services/                   # Live video feed inference, risk engine, SMS gateways, weather
│   ├── auth.py                     # JWT token authentication & bcrypt password hashing
│   ├── database.py                 # SQLite ORM engine & auto-loading environment configuration
│   ├── db_models.py                # SQLAlchemy database schema declarations
│   ├── schemas.py                  # Pydantic request/response validation contracts
│   ├── server.py                   # ASGI application entrypoint & video streaming endpoints
│   ├── requirements.txt            # Python dependencies
│   └── .env.example                # Backend environment template
├── frontend/                       # Vite + React Precision Farming Dashboard
│   ├── src/
│   │   ├── components/             # Reusable UI widgets (LiveFeedView, CropRiskBadge, Header, Modals)
│   │   ├── pages/                  # Route views (DashboardPage, SelectedFieldPage, AboutPage)
│   │   ├── services/               # REST API client services
│   │   └── App.jsx                 # Root layout & bilingual state provider
│   ├── package.json                # Frontend dependencies & scripts
│   └── vite.config.js              # Vite bundler configuration
├── survey/                         # Farmer Survey & Feedback Portal
│   ├── src/                        # Survey forms, interactive analytics & insights
│   ├── package.json                # Survey dependencies
│   └── vite.config.js              # Survey port & build configuration
├── FarmHawk_Project_Report.docx    # Full Academic & Technical Engineering Report
├── farmhawk_architecture.jpg       # System architecture diagram
├── farmhawk_economic_impact.png    # Economic impact chart
├── farmhawk_feasibility_radar.png  # Agronomic feasibility radar
├── requirements.txt                # Root Python dependencies
├── .env.example                    # Global environment template
├── LICENSE                         # MIT Open Source License
└── README.md                       # Master project documentation
```

---

## Quickstart Guide

### Prerequisites
- **Python 3.9 - 3.12**
- **Node.js 18+ & npm**
- **Git**
- Optional: NVIDIA GPU with CUDA for ultra-low latency YOLOv8 video inference

### 1. Clone the Repository
```bash
git clone https://github.com/RlyHarsh/Farmhawk.git
cd Farmhawk
```

### 2. Backend Setup
```bash
# Navigate to backend
cd backend

# Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your optional SMS gateway keys or keep defaults for local demo

# Launch FastAPI backend
uvicorn server:app --host 0.0.0.0 --port 5001 --reload
```
The backend API and interactive Swagger documentation will be live at:
- **API Documentation**: `http://localhost:5001/docs`
- **Health Check**: `http://localhost:5001/api/health`

### 3. Frontend Setup
```bash
# Navigate to frontend (in a separate terminal)
cd ../frontend

# Install dependencies
npm install

# Start Vite development server
npm run dev
```
The main FarmHawk web dashboard will open at `http://localhost:5173`.

### 4. Survey Portal Setup (Optional)
```bash
# Navigate to survey (in a separate terminal)
cd ../survey

# Install dependencies
npm install

# Start survey server
npm run dev
```
The farmer survey application will run at `http://localhost:5175`.

---

## 35-Class YOLOv8 Detection Taxonomy

FarmHawk's dual-stream model differentiates crop threats into two operational branches:

- **Blue Bounding Boxes — Insect Pests (6 Classes)**:
  `Rice Leaf Roller`, `Aphids`, `Fall Armyworm`, `Stem Borer`, `Whitefly`, `Legume Blister Beetle`.
- **Red Bounding Boxes — Foliar Pathogens & Diseases (29 Classes)**:
  `Wheat Yellow Rust`, `Wheat Leaf Rust`, `Corn Leaf Blight`, `Tomato Early Blight`, `Tomato Late Blight`, `Potato Late Blight`, `Powdery Mildew`, `Bacterial Blight`, and more.

---

## Configuration & Environment Variables

| Variable | Description | Default |
| :--- | :--- | :--- |
| `PORT` | FastAPI backend listening port | `5001` |
| `HOST` | Bind address | `0.0.0.0` |
| `SECRET_KEY` | JWT encryption secret key | Set in `.env` |
| `DEMO_FARMER_PHONE` | Default farmer mobile number for SMS dispatch | `+919876543210` |
| `TEXTBEE_API_KEY` | TextBee Android SIM Gateway API Key | Optional |
| `TEXTBEE_DEVICE_ID` | TextBee Connected Device ID | Optional |
| `VITE_API_BASE_URL` | Frontend API client base URL | `http://localhost:5001/api` |
| `VITE_SURVEY_URL` | Survey portal redirect URL | `http://localhost:5175` |

---

## REST API Reference

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/api/auth/login` | `POST` | Authenticates farmer and issues JSON Web Token. |
| `/api/fields` | `GET` | Retrieves all registered agricultural fields and health statuses. |
| `/api/fields/{id}/stream` | `GET` | MJPEG real-time video stream with live YOLOv8 inference overlays. |
| `/api/notifications/test-send` | `POST` | Dispatches bilingual verification alert via active SMS gateway. |
| `/api/notifications/simulate-shock` | `POST` | Simulates microclimate shocks (Heavy Rain, Heatwave) and triggers emergency SMS. |
| `/api/community/posts` | `GET/POST` | Krishi Samvad peer knowledge forum threads and expert advisories. |
| `/api/survey/submit` | `POST` | Submits farmer field feedback and validation responses. |

---

## Author & License

- **Lead Architect & Developer**: [Harshit Kushwaha](https://github.com/RlyHarsh)
- **License**: Released under the [MIT License](LICENSE).
