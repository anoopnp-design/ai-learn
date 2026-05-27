import pytest
from fastapi.testclient import TestClient
from src.app.main import app
from src.pipelines.preprocess import DataPreprocessingPipeline

def test_preprocessing_pipeline_dimensions():
    """Unit Test: Asserts that our pipeline maps a payload correctly."""
    pipeline = DataPreprocessingPipeline()
    mock_payload = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
    df = pipeline.payload_to_dataframe(mock_payload)
    scaled_vector = pipeline.scale_features_vectorized(df)
    assert scaled_vector.shape == (1, 4)

def test_predict_endpoint_success():
    """
    Integration Test: Asserts that hitting the live POST endpoint 
    passes model inference execution, printing the raw response on error.
    """
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    
    # Using context manager to force launch framework lifespan events
    with TestClient(app) as client:
        response = client.post("/api/v1/predict", json=payload)
        
        # --- RESILIENT DEBUGGING ASSERTION ---
        if response.status_code == 500:
            print(f"\n[CRITICAL ERROR LOG] Server responded with 500. Detail message: {response.text}")
            
        assert response.status_code == 200
        json_data = response.json()
        assert "predicted_class" in json_data
        assert json_data["status"] == "success"

def test_predict_endpoint_validation_error():
    """Contract Test: Asserts that out of bounds parameters trigger 422."""
    invalid_payload = {
        "sepal_length": -5.1, 
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    with TestClient(app) as client:
        response = client.post("/api/v1/predict", json=invalid_payload)
        assert response.status_code == 422