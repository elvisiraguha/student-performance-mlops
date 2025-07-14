#!/usr/bin/env python
# coding: utf-8

# #### Imports

# In[7]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

import mlflow
import mlflow.sklearn


# #### Load Data

# In[11]:


df = pd.read_csv("../data/student-mat.csv", sep=";")
binary_map = {"yes": 1, "no": 0}
binary_cols = ['schoolsup', 'famsup', 'paid', 'activities', 
               'nursery', 'higher', 'internet', 'romantic']
df[binary_cols] = df[binary_cols].map(lambda x: binary_map.get(x, x))


# #### Target and features

# In[12]:


target = "G3"
X = df.drop(columns=[target])
y = df[target]

numeric_cols = X.select_dtypes(include='number').columns.tolist()
categorical_cols = X.select_dtypes(include='object').columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_cols)
])


# #### Model configurations

# In[13]:


model_configs = {
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42)
}


# #### Start MLflow experiment

# In[15]:


mlflow.set_experiment("student-performance-regression")
mlflow.set_tracking_uri("http://127.0.0.1:5000")

for model_name, model in model_configs.items():
    with mlflow.start_run(run_name=model_name):
        # Build pipeline
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", model)
        ])

        # Fit and predict
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, preds)
        rmse = root_mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        # Log params and metrics
        mlflow.log_param("model", model_name)
        if hasattr(model, "n_estimators"):
            mlflow.log_param("n_estimators", model.n_estimators)

        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        # Log model artifact
        mlflow.sklearn.log_model(pipeline, "model")

        print(f"✅ Logged {model_name}: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.2f}")


# #### Register model

# In[29]:


from mlflow.tracking import MlflowClient

model_uri = f"runs:/{mlflow.last_active_run().info.run_id}/model"
model_name = "StudentPerformanceModel"

mlflow.register_model(model_uri=model_uri, name=model_name)

