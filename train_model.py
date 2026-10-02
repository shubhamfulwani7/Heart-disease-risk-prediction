import pandas as pd
import joblib
import json

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "heart.csv"

MODEL_DIR = BASE_DIR / "model"

MODEL_DIR.mkdir(exist_ok=True)


# ==========================================
# LOAD DATA
# ==========================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

df.columns = df.columns.str.strip()

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ==========================================
# CLEAN DATA
# ==========================================

print("\nMissing Values:")

print(df.isnull().sum())

df = df.dropna()

print("\nShape after cleaning:")

print(df.shape)


# ==========================================
# FEATURES / TARGET
# ==========================================

target = "target"

X = df.drop(columns=[target])

y = df[target]


print("\nFeatures:")

print(X.columns.tolist())


print("\nTarget Distribution:")

print(y.value_counts())


# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)


print("\nTraining Records:", len(X_train))

print("Testing Records:", len(X_test))


# ==========================================
# MODEL
# ==========================================

pipeline = Pipeline(

    steps=[

        (
            "scaler",

            StandardScaler()
        ),

        (
            "model",

            RandomForestClassifier(

                n_estimators=200,

                random_state=42,

                class_weight="balanced"

            )
        )

    ]

)


# ==========================================
# TRAIN
# ==========================================

print("\nTraining model...")

pipeline.fit(

    X_train,

    y_train

)


# ==========================================
# PREDICTION
# ==========================================

y_pred = pipeline.predict(X_test)


# ==========================================
# METRICS
# ==========================================

accuracy = accuracy_score(

    y_test,

    y_pred

)


precision = precision_score(

    y_test,

    y_pred,

    zero_division=0

)


recall = recall_score(

    y_test,

    y_pred,

    zero_division=0

)


f1 = f1_score(

    y_test,

    y_pred,

    zero_division=0

)


cm = confusion_matrix(

    y_test,

    y_pred

)


# ==========================================
# FEATURE IMPORTANCE
# ==========================================

rf_model = pipeline.named_steps["model"]

importance_values = rf_model.feature_importances_


feature_importance = {

    feature: float(importance)

    for feature, importance

    in zip(

        X.columns,

        importance_values

    )

}


# Sort descending

feature_importance = dict(

    sorted(

        feature_importance.items(),

        key=lambda item: item[1],

        reverse=True

    )

)


# ==========================================
# SAVE MODEL
# ==========================================

model_path = MODEL_DIR / "heart_model.pkl"

joblib.dump(

    pipeline,

    model_path

)


# ==========================================
# SAVE METRICS
# ==========================================

metrics = {

    "accuracy": round(

        float(accuracy) * 100,

        2

    ),

    "precision": round(

        float(precision) * 100,

        2

    ),

    "recall": round(

        float(recall) * 100,

        2

    ),

    "f1_score": round(

        float(f1) * 100,

        2

    ),

    "training_records": int(

        len(X_train)

    ),

    "testing_records": int(

        len(X_test)

    ),

    "total_records": int(

        len(df)

    ),

    "features": int(

        X.shape[1]

    ),

    "confusion_matrix": cm.tolist(),

    "feature_importance": feature_importance

}


metrics_path = MODEL_DIR / "metrics.json"


with open(

    metrics_path,

    "w"

) as file:

    json.dump(

        metrics,

        file,

        indent=4

    )


# ==========================================
# RESULTS
# ==========================================

print("\n================================")

print("MODEL RESULTS")

print("================================")

print(

    f"\nAccuracy: {accuracy * 100:.2f}%"

)

print(

    f"Precision: {precision * 100:.2f}%"

)

print(

    f"Recall: {recall * 100:.2f}%"

)

print(

    f"F1 Score: {f1 * 100:.2f}%"

)


print("\nConfusion Matrix:")

print(cm)


print("\nFeature Importance:")

for feature, importance in feature_importance.items():

    print(

        f"{feature}: "

        f"{importance:.4f}"

    )


print("\n================================")

print("SUCCESS")

print("================================")

print("\nModel saved:")

print(model_path)

print("\nMetrics saved:")

print(metrics_path)