import os
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from src.pipelines.preprocess import DataPreprocessingPipeline

def run_training_pipeline():
    print("[Training Pipeline] Starting model training lifecycle...")
    
    # 1. Fetching historical training data
    print("[Training Pipeline] Loading historical dataset...")
    raw_data = load_iris(as_frame=True)
    X = raw_data.data  
    y = raw_data.target  
    
    # Map the Scikit-Learn naming convention to our clean enterprise pipeline keys
    column_mapping = {
        "sepal length (cm)": "sepal_length",
        "sepal width (cm)": "sepal_width",
        "petal length (cm)": "petal_length",
        "petal width (cm)": "petal_width"
    }
    X = X.rename(columns=column_mapping)

    # 2. Train/Test Split (Strictly checking parameters)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Apply Data Normalization via our Pipeline Service
    print("[Training Pipeline] Normalizing training data matrices...")
    pipeline = DataPreprocessingPipeline()
    X_train_scaled = pipeline.scale_features_vectorized(X_train)
    X_test_scaled = pipeline.scale_features_vectorized(X_test)
    
    # 4. Initialize and Fit the Model Architecture
    print("[Training Pipeline] Fitting mathematical model weights...")
    model = LogisticRegression(max_iter=200)
    model.fit(X_train_scaled, y_train)
    
    # 5. Evaluate Accuracy Performance
    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    print(f"[Training Pipeline] Model Evaluation Complete. Test Accuracy: {accuracy * 100:.2f}%")
    
    # 6. Serialize and Persist the Binary State to Disk
    os.makedirs("artifacts", exist_ok=True) # Ensures folder exists like Directory.CreateDirectory()
    artifact_path = "artifacts/iris_classifier_v1.pkl"
    print(f"[Training Pipeline] Exporting immutable model artifact to: {artifact_path}")
    joblib.dump(model, artifact_path)
    print("[Training Pipeline] Optimization complete.")

if __name__ == "__main__":
    run_training_pipeline()