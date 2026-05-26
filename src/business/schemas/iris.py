from pydantic import BaseModel, Field

class IrisInputDTO(BaseModel):
    sepal_length: float = Field(..., description="Length of the sepal in cm", ge=0.1, le=10.0)
    sepal_width: float = Field(..., description="Width of the sepal in cm", ge=0.1, le=10.0)
    petal_length: float = Field(..., description="Length of the petal in cm", ge=0.1, le=10.0)
    petal_width: float = Field(..., description="Width of the petal in cm", ge=0.1, le=10.0)

    # Provide a payload template for Swagger documentation
    model_config = {
        "json_schema_extra": {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }
    }

class IrisPredictionOutputDTO(BaseModel):
    predicted_class: int
    model_name: str
    status: str