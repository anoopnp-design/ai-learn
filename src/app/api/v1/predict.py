from fastapi import APIRouter,HTTPException
from src.business.schemas.iris import IrisInputDTO,IrisPredictionOutputDTO
from src.app.main import app_state
from src.pipelines.preprocess import DataPreprocessingPipeline

router = APIRouter();

# Instantiate our pipeline component
preprocess_pipeline = DataPreprocessingPipeline()



@router.post("/predict",response_model=IrisPredictionOutputDTO)
async def predit_iris_class(payload:IrisInputDTO):
    
    # Verify our "Model" asset is loaded in the global memory state
    if "ml_model" not in app_state:
        raise HTTPException(status_code=500, detail="ML Model state engine not initialized.")
    
    # 1. Convert DTO to standard native dictionary format
    raw_data = payload.model_dump()
    # 2. Extract and scale vectors via the pipeline service
    features_dataframe = preprocess_pipeline.payload_to_dataframe(raw_data)
    scaled_vector = preprocess_pipeline.scale_features_vectorized(features_dataframe)

    # Print out matrix logs to our container terminal standard output for tracing
    print(f"Executing inference calculation on input vector array: {scaled_vector}")

    # For now, we mock the execution prediction loop before training our real model
    # A true .NET architect always mocks integrations first!
    mock_prediction = 0 # Class 0 = Setosa
    
    return IrisPredictionOutputDTO(
        predicted_class=mock_prediction,
        model_name=app_state["ml_model"]["model_name"],
        status=app_state["ml_model"]["status"]
    )
