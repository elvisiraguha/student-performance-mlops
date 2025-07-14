# 🎓 Student Performance Prediction - MLOps Zoomcamp Project

This project is part of the [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) and demonstrates a complete MLOps workflow for building, deploying, and monitoring a machine learning model.

The goal is to predict the final grade (G3) of students in a math course using a dataset of demographic, academic, and social features.

---

## 📊 Dataset

The dataset comes from the UCI Machine Learning Repository and contains various student attributes:

- **Demographic**: age, sex, address type
- **Social**: family size, parental status, romantic relationships
- **Academic**: study time, failures, previous grades (G1, G2), support classes
- **Target**: `G3` - final math grade (0–20)

Data source: [UCI Student Performance Data Set](https://archive.ics.uci.edu/ml/datasets/Student+Performance)

---

## ⚙️ Project Structure


├── data/ # Raw and processed data

├── notebooks/ # EDA, experimentation and training

  └── eda.ipynb # Exploratory Data Analysis

  └── eda.py # Exploratory Data Analysis Script

  └── preprocessing_and_modeling_with_mlflow.ipynb

  └── preprocessing_and_modeling_with_mlflow.py

├── serving/ # Model serving with FastAPI

  └── main.py # FastAPI app for model inference

├── requirements.txt # Python dependencies

├── README.md # Project overview

---

## 🚀 MLOps Stack

- **Experiment Tracking**: MLflow
- **Model Training**: scikit-learn, pandas
- **Model Serving**: FastAPI

---

## 🛠️ Setup Instructions

### Clone the repo

Clone [https://github.com/elvisiraguha/student-performance-mlops](https://github.com/elvisiraguha/student-performance-mlops)

```
git clone https://github.com/elvisiraguha/student-performance-mlops.git
cd student-performance-mlops
```

### Create virtual environment

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run training

```
python notebooks/preprocessing_and_modeling_with_mlflow.py
```

### 3. 🚀 Launch MLflow UI

```
mlflow ui
```

Visit: http://localhost:5000

- View experiment runs

- Compare metrics (MAE, RMSE, R²)

- Manage registered models (Staging/Production)


## 🧠 Model Serving via FastAPI
We serve the registered model using FastAPI + MLflow model loading.
### Run the API server

```
uvicorn serving.main:app --reload
```

## 📈 Sample Prediction Request

```
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "school": "GP", "sex": "F", "age": 17, "address": "U", "famsize": "GT3",
        "Pstatus": "T", "Medu": 4, "Fedu": 4, "Mjob": "teacher", "Fjob": "services",
        "reason": "course", "guardian": "mother", "traveltime": 1, "studytime": 2,
        "failures": 0, "schoolsup": 0, "famsup": 1, "paid": 0, "activities": 1,
        "nursery": 1, "higher": 1, "internet": 1, "romantic": 0, "famrel": 5,
        "freetime": 3, "goout": 3, "Dalc": 1, "Walc": 2, "health": 5, "absences": 3,
        "G1": 14, "G2": 15
      }'

```

## ✅ Features Completed
- Data cleaning and preprocessing

- EDA and feature exploration

- Model training and evaluation

- MLflow experiment logging

- Model registration to Model Registry

- Model serving via FastAPI

## 📦 Next Steps
- Dockerize training and serving

- CI/CD for model updates

- Monitoring with Prometheus + Grafana

- Deploy to cloud (e.g., GCP, AWS, or Azure)

## 🧑‍💻 Author
Elvis Iraguha – @elvisiraguha

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
