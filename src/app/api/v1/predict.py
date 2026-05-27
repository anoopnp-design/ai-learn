from fastapi import APIRouter, HTTPException
from src.business.schemas.iris import IrisInputDTO, IrisPredictionOutputDTO
from src.app.main import app_state
from src.pipelines.preprocess import DataPreprocessingPipeline

router = APIRouter()
preprocess_pipeline = DataPreprocessingPipeline()

# Define the prediction category mapping string values
CLASS_MAPPING = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

@router.post("/predict", response_model=IrisPredictionOutputDTO)
def predict_iris_class(payload: IrisInputDTO):  # <-- REMOVED 'async' to run safely on background threads
    """
    Production Inference Endpoint. 
    Ingests validation DTO payloads, runs vector scaling normalization,
    and returns a live prediction calculated by our underlying AI engine binary.
    """
    # 1. Thread-Safe Dependency Check
    model = app_state.get("ml_model")
    if model is None:
        raise HTTPException(status_code=500, detail="Inference engine binary state is uninitialized.")
    
    try:
        # 2. Extract Data Payload and pipeline process
        raw_dict = payload.model_dump()
        df_features = preprocess_pipeline.payload_to_dataframe(raw_dict)
        scaled_matrix = preprocess_pipeline.scale_features_vectorized(df_features)
        
        # 3. Compute Real Live Inference 
        # model.predict() outputs a NumPy array containing the targeted class index (e.g., array([0]))
        prediction_array = model.predict(scaled_matrix)
        predicted_index = int(prediction_array[0])
        
        # 4. Map index to explicit human readable string names
        flower_name = CLASS_MAPPING.get(predicted_index, "Unknown Class")
        
        return IrisPredictionOutputDTO(
            predicted_class=predicted_index,
            model_name=f"Iris Logistic Regression Classifier - {flower_name}",
            status="success"
        )
        
    except Exception as ex:
        # Map unexpected failure frames to an explicit internal server log response
        print(f"[Inference Error] Exception encountered during evaluation pipeline: {str(ex)}")
        raise HTTPException(status_code=500, detail="Internal inference execution exception encountered.")