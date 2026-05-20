# ⚡ PHYSI-CAST QUICK REFERENCE CARD

**Save this file for quick reference!**

---

## 🚀 QUICK START (Copy & Paste)

### Windows Users

```bash
# 1. Setup (one time)
cd "c:\Users\vns75\OneDrive\Desktop\Edi 2\PINN-Climate-App"
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 2. Train (20 min)
python main.py

# 3. Start API (Terminal 1)
python -m uvicorn api.server:app --reload --port 8000

# 4. Start Dashboard (Terminal 2)
streamlit run dashboard/app.py

# 5. Open browser
http://localhost:8501
```

### Mac/Linux Users

```bash
# 1. Setup (one time)
cd "c:\Users\vns75\OneDrive\Desktop\Edi 2\PINN-Climate-App"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Train (20 min)
python main.py

# 3. Start API (Terminal 1)
python -m uvicorn api.server:app --reload --port 8000

# 4. Start Dashboard (Terminal 2)
streamlit run dashboard/app.py

# 5. Open browser
http://localhost:8501
```

---

## 📚 DOCUMENTATION FILES

| File | Read This For |
|------|---|
| **GETTING_STARTED.md** | First time setup |
| **README.md** | Complete reference |
| **MANIFEST.md** | Project structure |
| **PROJECT_SUMMARY.md** | Full overview |
| **BUILD_COMPLETE.md** | What was built |

---

## 🎯 COMMON COMMANDS

### Training
```bash
# Quick training (5 min)
python main.py --n-train-samples 2000 --adam-epochs 50

# Standard training (20 min)
python main.py

# High accuracy (60+ min)
python main.py --n-train-samples 10000 --adam-epochs 200

# With GPU
python main.py --device cuda
```

### Running Components
```bash
# Train model
python main.py

# Start API
python -m uvicorn api.server:app --reload --port 8000

# Start Dashboard
streamlit run dashboard/app.py

# Run tests
python test_components.py --all

# Check dependencies
python test_components.py --check
```

### Using Helper Scripts (Windows)
```bash
run.bat check        # Check dependencies
run.bat train        # Train model
run.bat api          # Start API
run.bat dashboard    # Start dashboard
run.bat test         # Run tests
```

### Using Helper Scripts (Mac/Linux)
```bash
./run.sh check       # Check dependencies
./run.sh train       # Train model
./run.sh api         # Start API
./run.sh dashboard   # Start dashboard
./run.sh test        # Run tests
```

---

## 🌐 ACCESS POINTS

| Service | URL | Port |
|---------|-----|------|
| Dashboard | http://localhost:8501 | 8501 |
| API | http://localhost:8000 | 8000 |
| API Docs | http://localhost:8000/docs | 8000 |
| Health Check | http://localhost:8000/health | 8000 |

---

## 🔧 API ENDPOINTS

### Get Prediction
```bash
curl -X POST "http://localhost:8000/predict/point" \
  -H "Content-Type: application/json" \
  -d '{"x": 40.7128, "y": -74.0060, "z": 10, "t": 3600}'
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

### Get Alerts
```bash
curl -X POST "http://localhost:8000/alerts/extreme-weather" \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 40.7128,
    "longitude": -74.0060,
    "altitude": 0
  }'
```

### Check Health
```bash
curl http://localhost:8000/health
```

---

## 📊 DASHBOARD FEATURES

### Current Conditions Tab 📍
- View weather at specific location
- Temperature, pressure, wind metrics
- Confidence scoring
- Wind vector visualization

### Spatial Grid Tab 🗺️
- Interactive map
- 10×10 grid predictions
- Temperature heatmap
- Wind patterns

### Alerts Tab ⚠️
- Extreme weather detection
- Recommendations
- Confidence levels
- Alert types

### Analytics Tab 📈
- 24-72 hour forecasts
- Temperature trends
- Wind patterns
- Statistics

---

## 🐛 TROUBLESHOOTING

### Issue: ModuleNotFoundError
```bash
# Solution: Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Issue: Port Already in Use
```bash
# Solution: Use different port
python -m uvicorn api.server:app --port 8001
```

### Issue: GPU Not Found
```bash
# Solution: Check CUDA installation or use CPU
python main.py --device cpu
```

### Issue: Out of Memory
```bash
# Solution: Reduce batch size and data
python main.py --n-train-samples 2000 --batch-size 16
```

---

## 📁 KEY DIRECTORIES

| Directory | Purpose |
|-----------|---------|
| `src/` | Core ML modules |
| `api/` | FastAPI backend |
| `dashboard/` | Streamlit UI |
| `models/` | Model checkpoints |
| `results/` | Training results |
| `data/` | Training data |

---

## 🔑 KEY FILES

| File | What It Does |
|------|---|
| `main.py` | Train the model |
| `config.py` | Customize settings |
| `src/physics.py` | Physics constraints |
| `src/network.py` | Neural network |
| `src/trainer.py` | Training pipeline |
| `api/server.py` | REST API |
| `dashboard/app.py` | Web dashboard |

---

## 💾 SAVING/LOADING MODELS

### Save Model
```python
import torch
torch.save(model.network.state_dict(), 'my_model.pth')
```

### Load Model
```python
import torch
model.network.load_state_dict(torch.load('my_model.pth'))
```

---

## 🎓 TRAINING PARAMETERS

```
--n-train-samples INT        [default: 5000]    Training samples
--n-collocation INT          [default: 5000]    Collocation points
--batch-size INT             [default: 32]      Batch size
--adam-epochs INT            [default: 100]     Adam epochs
--learning-rate FLOAT        [default: 0.001]   Adam LR
--lbfgs-iter INT            [default: 50]      L-BFGS iterations
--lambda-data FLOAT          [default: 0.5]     Data loss weight
--lambda-physics FLOAT       [default: 0.5]     Physics loss weight
```

---

## ⏱️ EXPECTED TIMES

| Task | GPU | CPU |
|------|-----|-----|
| Quick Train | 5 min | 30 min |
| Standard Train | 20 min | 2-3 hrs |
| High Accuracy | 60+ min | 6+ hrs |
| Single Prediction | <10ms | <50ms |
| Grid (10×10) | <100ms | <500ms |

---

## 🔄 TYPICAL WORKFLOW

1. **Install** → pip install -r requirements.txt
2. **Test** → python test_components.py --all
3. **Train** → python main.py
4. **API** → python -m uvicorn api.server:app --reload
5. **Dashboard** → streamlit run dashboard/app.py
6. **Access** → http://localhost:8501
7. **Predict** → Use dashboard or API
8. **Export** → Save results as CSV

---

## 📞 HELP RESOURCES

- **Quick Start**: GETTING_STARTED.md
- **Full Reference**: README.md
- **Structure**: MANIFEST.md
- **Tests**: python test_components.py --help
- **API Docs**: http://localhost:8000/docs
- **Code Comments**: Check source files in src/

---

## ✅ CHECKLIST FOR FIRST USE

- [ ] Python 3.10+ installed
- [ ] Dependencies installed
- [ ] Tests passing
- [ ] Model training started
- [ ] API running on port 8000
- [ ] Dashboard running on port 8501
- [ ] Dashboard accessible at localhost:8501
- [ ] Made first prediction
- [ ] Viewed some alerts
- [ ] Explored analytics tab

---

## 🎁 BONUS TIPS

1. **Faster Training**: Use GPU with CUDA
2. **Quick Demo**: Use --n-train-samples 1000
3. **Production**: Train with full dataset
4. **Customization**: Edit config.py before training
5. **API Integration**: Use requests library in Python
6. **Dashboard**: Can be deployed publicly with ngrok
7. **Monitoring**: Watch training with nvidia-smi
8. **Backup**: Save models regularly

---

## 🆘 EMERGENCY HELP

### Can't install packages
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

### Port conflicts
```bash
# Find what's using port 8000/8501
lsof -i :8000
# Kill it
kill -9 <PID>
```

### Virtual environment issues
```bash
# Remove and recreate
rm -rf venv
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

## 📱 QUICK LINKS

- 📖 [README.md](README.md) - Full documentation
- 🚀 [GETTING_STARTED.md](GETTING_STARTED.md) - Setup guide  
- 📋 [MANIFEST.md](MANIFEST.md) - File reference
- 🎯 [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Overview
- ✅ [BUILD_COMPLETE.md](BUILD_COMPLETE.md) - What's included

---

<div align="center">

**PHYSI-CAST - QUICK REFERENCE**

*Physics-Informed Climate Forecasting*

---

**Start**: Read GETTING_STARTED.md  
**Run**: `python main.py` then `run.bat api` then `run.bat dashboard`  
**Access**: http://localhost:8501  
**API Docs**: http://localhost:8000/docs

---

Version 1.0.0 | Ready to Use ✅

</div>
