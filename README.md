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

├── notebooks/ # EDA and experimentation

├── training/ # Training pipeline

├── serving/ # Model serving with FastAPI

├── monitoring/ # Monitoring with Prometheus & Grafana

├── Dockerfile # Image for training and serving

├── requirements.txt # Python dependencies

└── README.md # Project overview


---

## 🚀 MLOps Stack

- **Experiment Tracking**: MLflow
- **Model Training**: scikit-learn, pandas
- **Model Serving**: FastAPI
- **Orchestration**: Prefect / Airflow (optional)
- **Containerization**: Docker
- **Monitoring**: Prometheus, Grafana
- **Deployment**: Docker Compose

---

## 🛠️ Setup Instructions

### Clone the repo

```
git clone [https://github.com/<your-username>/student-performance-mlops](https://github.com/elvisiraguha/student-performance-mlops).git
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
python training/train.py
```

### Run the API server

```
uvicorn serving.main:app --reload
```

## 📈 Sample Prediction Request

```
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "school": "GP",
           "sex": "F",
           "age": 17,
           "address": "U",
           "famsize": "GT3",
           ...
           "G1": 14,
           "G2": 15
         }'
```

## 📉 Evaluation

Model performance is tracked using:

- MAE / RMSE scores

- MLflow for experiment comparison

- Monitoring with Prometheus and Grafana dashboards

## 🧠 Key MLOps Concepts Covered

- Data versioning

- Experiment tracking

- Containerized training and inference

- API-based model serving

- Continuous monitoring

## 🧑‍💻 Author
Elvis Iraguha – @elvisiraguha

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
