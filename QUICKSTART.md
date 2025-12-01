# Quick Start Guide

## Setup

1. **Create Virtual Environment**
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. **Install Dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Configure System**
   ```powershell
   copy config\config.example.yaml config\config.yaml
   copy .env.example .env
   # Edit config.yaml and .env with your settings
   ```

## Quick Demo

### 1. Generate Sample Data
```powershell
python scripts\generate_sample_data.py
```

### 2. Train Model
```powershell
python examples\train_example.py
```

### 3. Run Prediction
```powershell
python src\predict.py --equipment-id EQ001
```

### 4. Start API Server
```powershell
python src\api\main.py
```

### 5. Test API (in another terminal)
```powershell
python examples\api_example.py
```

## Project Structure

```
predictive maintenance ai/
├── config/              # Configuration files
├── data/               # Data storage
│   ├── raw/           # Raw sensor data
│   ├── processed/     # Processed data
│   └── sample_sensor_data.csv
├── examples/          # Usage examples
├── models/            # Saved models
│   └── trained/      # Trained model files
├── scripts/           # Utility scripts
├── src/              # Source code
│   ├── api/          # REST API
│   ├── data_processing/  # Data preprocessing
│   ├── features/     # Feature engineering
│   ├── models/       # ML models
│   ├── monitoring/   # Logging & alerts
│   └── utils/        # Utilities
├── tests/            # Unit tests
├── logs/             # Log files
├── .env              # Environment variables
└── requirements.txt  # Dependencies
```

## Next Steps

1. Review and customize `config/config.yaml`
2. Set up database connection (PostgreSQL recommended)
3. Train models with your own data
4. Deploy API to production
5. Set up monitoring and alerts
