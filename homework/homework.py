#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#

import gzip
import json
import os
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import median_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# ---------------------------------------------------------------------------
# Paso 1: Preprocesamiento
# ---------------------------------------------------------------------------

def load_and_preprocess(path):
    df = pd.read_csv(path)
    df["Age"] = 2021 - df["Year"]
    df = df.drop(columns=["Year", "Car_Name"])
    return df


train_df = load_and_preprocess("files/input/train_data.csv.zip")
test_df = load_and_preprocess("files/input/test_data.csv.zip")

# ---------------------------------------------------------------------------
# Paso 2: División en X / y
# ---------------------------------------------------------------------------

x_train = train_df.drop(columns=["Present_Price"])
y_train = train_df["Present_Price"]

x_test = test_df.drop(columns=["Present_Price"])
y_test = test_df["Present_Price"]

# ---------------------------------------------------------------------------
# Paso 3: Pipeline
# ---------------------------------------------------------------------------

categorical_cols = ["Fuel_Type", "Selling_type", "Transmission"]
numerical_cols = [c for c in x_train.columns if c not in categorical_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numerical_cols),
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("scaler", MinMaxScaler()),
        ("selector", SelectKBest(score_func=f_regression)),
        ("regressor", LinearRegression()),
    ]
)

# ---------------------------------------------------------------------------
# Paso 4: Optimización de hiperparámetros con validación cruzada
# ---------------------------------------------------------------------------

total_features = len(categorical_cols) * 2 + len(numerical_cols)  # rough upper bound
param_grid = {
    "selector__k": list(range(1, total_features + 2)),
}

model = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=10,
    scoring="neg_mean_absolute_error",
    refit=True,
    n_jobs=-1,
)

model.fit(x_train, y_train)

# ---------------------------------------------------------------------------
# Paso 5: Guardar modelo
# ---------------------------------------------------------------------------

os.makedirs("files/models", exist_ok=True)

with gzip.open("files/models/model.pkl.gz", "wb") as f:
    pickle.dump(model, f)

# ---------------------------------------------------------------------------
# Paso 6: Métricas
# ---------------------------------------------------------------------------

os.makedirs("files/output", exist_ok=True)

metrics_rows = []
for split, X, y in [("train", x_train, y_train), ("test", x_test, y_test)]:
    y_pred = model.predict(X)
    metrics_rows.append(
        {
            "type": "metrics",
            "dataset": split,
            "r2": round(r2_score(y, y_pred), 3),
            "mse": round(mean_squared_error(y, y_pred), 3),
            "mad": round(median_absolute_error(y, y_pred), 3),
        }
    )

with open("files/output/metrics.json", "w", encoding="utf-8") as f:
    for row in metrics_rows:
        f.write(json.dumps(row) + "\n")

print("Modelo entrenado. Mejores parámetros:", model.best_params_)
print("Métricas:")
for row in metrics_rows:
    print(row)
