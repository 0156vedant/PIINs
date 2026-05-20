# 🚀 Getting Started with Physi-Cast

Welcome! This guide will help you get **Physi-Cast** up and running in minutes.

---

## ⚡ Quick Start (5 Minutes)

### 1. Install Dependencies

```bash
# Navigate to project directory
cd "c:\Users\vns75\OneDrive\Desktop\Edi 2\PINN-Climate-App"

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### 2. Run Tests

```bash
# Test all components
python test_components.py --all

# Or using the helper script:
# Windows:
run.bat test

# Mac/Linux:
./run.sh test
```

### 3. Train the Model

```bash
# Quick training (5 minutes on GPU)
python main.py --n-train-samples 2000 --adam-epochs 50 --lbfgs-iter 20

# Or standard training (20 minutes on GPU)
python main.py

# Or using the helper script:
run.bat train    # Windows
./run.sh train   # Mac/Linux
```

### 4. Start the API Backend

```bash
# Terminal 1 - Start API
python -m uvicorn api.server:app --reload --port 8000

# Or:
run.bat api      # Windows
./run.sh api     # Mac/Linux

# You should see:
# Uvicorn running on http://0.0.0.0:8000
```

### 5. Start the Dashboard

```bash
# Terminal 2 - Start Dashboard
streamlit run dashboard/app.py

# Or:
run.bat dashboard   # Windows
./run.sh dashboard  # Mac/Linux

# You should see:
# Local URL: http://localhost:8501
```

### 6. Open the Dashboard

Go to your browser and visit:
- **Dashboard**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs

---

## 📖 Detailed Setup

### Prerequisites

- **Python 3.10+**
- **pip** (Python package manager)
- **git** (optional, for version control)
- **4GB+ RAM** (8GB+ recommended)
- **GPU with CUDA** (optional, for faster training)

### Installation Steps

#### Step 1: Clone or Navigate to Project

```bash
cd "c:\Users\vns75\OneDrive\Desktop\Edi 2\PINN-Climate-App"
```

#### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

#### Step 3: Upgrade pip

```bash
python -m pip install --upgrade pip
```

#### Step 4: Install Requirements

```bash
pip install -r requirements.txt
```

This installs:
- PyTorch (Deep Learning)
- FastAPI (API Backend)
- Streamlit (Web Dashboard)
- NumPy, SciPy (Scientific Computing)
- Plotly, Folium (Visualization)
- And more...

#### Step 5: Verify Installation

```bash
# Test components
python test_components.py --all

# You should see: "Total: 5/5 tests passed"
```

---

## 🎓 Training the Model

### Basic Training

```bash
python main.py
```

This will:
1. Generate synthetic training data (5000 samples)
2. Generate collocation points (5000 points)
3. Train using Adam optimizer (100 epochs)
4. Fine-tune using L-BFGS (50 iterations)
5. Save model to `models/pinn_model.pth`

**Estimated Time**: 20-30 minutes on GPU, 2-3 hours on CPU

### Quick Training (For Testing)

```bash
python main.py \
    --n-train-samples 2000 \
    --n-collocation 2000 \
    --adam-epochs 50 \
    --lbfgs-iter 20
```

**Estimated Time**: 5-10 minutes on GPU

### High-Accuracy Training

```bash
python main.py \
    --n-train-samples 10000 \
    --n-collocation 10000 \
    --adam-epochs 200 \
    --lbfgs-iter 100 \
    --lambda-physics 0.6
```

**Estimated Time**: 60+ minutes on GPU

### Monitor Training

During training, you'll see:
```
[STAGE 1] Adam Optimizer (Initial Convergence)
Epoch 1: total_loss=0.123456, data_loss=0.087654, physics_loss=0.035802
Epoch 2: total_loss=0.102345, data_loss=0.072345, physics_loss=0.029999
...
[STAGE 2] L-BFGS Optimizer (Physics Refinement)
L-BFGS Iter 1: Loss = 9.876543e-03
L-BFGS Iter 2: Loss = 8.765432e-03
...
```

---

## 🌐 Running the System

### Architecture

```
┌────────────────────────────────────────┐
│   Streamlit Dashboard (Port 8501)      │
│   Beautiful web UI for predictions     │
└──────────────────┬─────────────────────┘
                   │
              HTTP Requests
                   │
┌──────────────────▼─────────────────────┐
│   FastAPI Backend (Port 8000)          │
│   REST API with Swagger docs           │
└──────────────────┬─────────────────────┘
                   │
            PyTorch Inference
                   │
┌──────────────────▼─────────────────────┐
│   PINN Model                           │
│   Physics-Informed Neural Network      │
└────────────────────────────────────────┘
```

### Running Individual Components

#### Only API Backend

```bash
python -m uvicorn api.server:app --reload --port 8000

# Access API docs at: http://localhost:8000/docs
```

#### Only Dashboard

```bash
streamlit run dashboard/app.py

# Access dashboard at: http://localhost:8501
```

#### Everything Together

```bash
# Terminal 1
python -m uvicorn api.server:app --reload --port 8000

# Terminal 2 (while Terminal 1 is running)
streamlit run dashboard/app.py
```

---

## 📊 Dashboard Tour

### Current Conditions Tab 📍
- Real-time weather prediction at a specific location
- Temperature, pressure, wind speed, wind components
- Wind vector visualization
- Model confidence score

**Features:**
- Input location (latitude/longitude/altitude)
- Real-time weather metrics
- Physics validation indicators

### Spatial Grid Tab 🗺️
- Interactive map showing forecast area
- 10×10 grid predictions
- Temperature heatmap
- Wind field visualization

**Features:**
- Click "Generate Grid Predictions" to see spatial distribution
- Hover on map to see precise coordinates
- Zoom and pan the map

### Alerts & Warnings Tab ⚠️
- Automated extreme weather detection
- Alerts for frost, heat waves, severe winds
- Actionable recommendations
- Confidence scores

**Alert Types:**
- 🔴 **CRITICAL**: Extreme conditions
- 🟠 **HIGH**: Severe conditions
- 🟡 **MEDIUM**: Moderate conditions
- 🟢 **LOW**: Minor conditions

### Analytics Tab 📈
- 24-72 hour forecast trends
- Temperature trends
- Wind speed patterns
- Statistical summaries

**Visualizations:**
- Line charts for temperature/wind
- Forecast statistics
- Anomaly indicators

---

## 🔌 API Endpoints

### Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

### Predict Weather at Point

```bash
curl -X POST "http://localhost:8000/predict/point" \
  -H "Content-Type: application/json" \
  -d '{
    "x": 40.7128,
    "y": -74.0060,
    "z": 10,
    "t": 3600
  }'
```

### Get Grid Predictions

```bash
curl -X POST "http://localhost:8000/predict/grid" \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 40.7128,
    "longitude": -74.0060,
    "altitude": 0,
    "forecast_hours": 24
  }'
```

### Detect Extreme Weather

```bash
curl -X POST "http://localhost:8000/alerts/extreme-weather" \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 40.7128,
    "longitude": -74.0060,
    "altitude": 0
  }'
```

**Complete API documentation**: http://localhost:8000/docs

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'torch'"

**Solution:**
```bash
# Reinstall PyTorch
pip uninstall torch
pip install torch

# Or with GPU support:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Port 8000 Already in Use

**Solution:**
```bash
# Find process using port 8000 and kill it
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Mac/Linux:
lsof -i :8000
kill -9 <PID>

# Or use different port:
python -m uvicorn api.server:app --port 8001
```

### Issue: Dashboard Won't Load

**Solution:**
```bash
# Clear Streamlit cache
streamlit cache clear

# Reinstall Streamlit
pip uninstall streamlit
pip install streamlit

# Run with verbose logging
streamlit run dashboard/app.py --logger.level=debug
```

### Issue: Training is Very Slow

**Solution:**
1. **Use GPU**: Install CUDA for PyTorch
2. **Reduce data size**: `--n-train-samples 2000`
3. **Reduce epochs**: `--adam-epochs 50`

---

## 📚 Next Steps

### Explore the Code

```
src/
├── physics.py     # Understand PDE constraints
├── network.py     # Learn FCNN architecture
├── trainer.py     # Study optimization pipeline
└── utils.py       # Data processing utilities
```

### Customize Configuration

Edit `config.py` to modify:
- Neural network architecture
- Training hyperparameters
- Physics constraint weights
- API and dashboard settings

### Deploy in Production

1. Use a production WSGI server (gunicorn)
2. Set up HTTPS with certificate
3. Use environment variables for secrets
4. Deploy with Docker or cloud platform

### Integrate with External Data

Modify `src/utils.py` to:
- Fetch real ERA5 data from ECMWF
- Integrate IoT sensor networks
- Connect to weather APIs
- Load custom datasets

---

## 💡 Tips & Tricks

### Faster Training on CPU

```bash
# Use reduced configuration
python main.py \
    --n-train-samples 1000 \
    --adam-epochs 25 \
    --lbfgs-iter 10
```

### Monitor GPU Usage

```bash
# Windows PowerShell
nvidia-smi -l 1  # Refresh every 1 second

# Mac/Linux
watch nvidia-smi
```

### Save/Load Custom Models

```python
# Save
torch.save(model.network.state_dict(), 'my_model.pth')

# Load
model.network.load_state_dict(torch.load('my_model.pth'))
```

### Export Results as CSV

```python
# From API or dashboard
import pandas as pd

# Save predictions
df = pd.DataFrame(predictions)
df.to_csv('forecast.csv', index=False)
```

---

## 🎓 Learning Resources

### Physics-Informed Neural Networks
- Paper: [Physics-Informed Neural Networks (Raissi et al., 2019)](https://arxiv.org/abs/1711.10566)
- Tutorial: [PINN Basics](https://deepxde.readthedocs.io/)

### Deep Learning
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Fast.ai Course](https://course.fast.ai/)

### Climate Science
- [ECMWF Weather Models](https://www.ecmwf.int/)
- [NOAA Climate Predictions](https://www.noaa.gov/)

---

## ✅ Checklist

- [ ] Python 3.10+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Tests pass (`python test_components.py --all`)
- [ ] Model trained (`python main.py`)
- [ ] API running (`python -m uvicorn api.server:app --reload`)
- [ ] Dashboard running (`streamlit run dashboard/app.py`)
- [ ] Dashboard accessible at `http://localhost:8501`
- [ ] API docs accessible at `http://localhost:8000/docs`

---

## 🎉 Success!

You're now ready to use **Physi-Cast**! 

**Next steps:**
1. Explore the dashboard
2. Make predictions
3. Check generated alerts
4. Review the analytics
5. Customize for your use case

**Questions?** Check the [README.md](README.md) or see the [API documentation](http://localhost:8000/docs).

---

<div align="center">

**Happy Forecasting! 🌤️**

*Physics-Informed Climate Intelligence at Your Fingertips*

</div>
