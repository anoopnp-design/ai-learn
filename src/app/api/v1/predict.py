from fastapi import APIRouter, HTTPException, Request
from src.business.schemas.iris import IrisInputDTO, IrisPredictionOutputDTO
from src.pipelines.preprocess import DataPreprocessingPipeline

router = APIRouter()
preprocess_pipeline = DataPreprocessingPipeline()

CLASS_MAPPING = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

@router.post("/predict", response_model=IrisPredictionOutputDTO)
def predict_iris_class(payload: IrisInputDTO, request: Request):  # <-- ADDED request parameter
    """
    Production Inference Endpoint.
    Extracts the model from the Request context state safely and predictably.
    """
    # Core Architecture Fix: Pull model from the specific active request state engine
    model = getattr(request.app.state, "ml_model", None)
    
    if model is None:
        raise HTTPException(
            status_code=500, 
            detail="Inference engine binary state is uninitialized inside request context."
        )
    
    try:
        raw_dict = payload.model_dump()
        df_features = preprocess_pipeline.payload_to_dataframe(raw_dict)
        scaled_matrix = preprocess_pipeline.scale_features_vectorized(df_features)
        
        prediction_array = model.predict(scaled_matrix)
        predicted_index = int(prediction_array[0])
        flower_name = CLASS_MAPPING.get(predicted_index, "Unknown Class")
        
        return IrisPredictionOutputDTO(
            predicted_class=predicted_index,
            model_name=f"Iris Logistic Regression Classifier - {flower_name}",
            status="success"
        )
        
    except Exception as ex:
        print(f"[Inference Error] Exception: {str(ex)}")
        raise HTTPException(status_code=500, detail="Internal inference execution exception encountered.")