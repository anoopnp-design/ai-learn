import joblib
from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.app.core.config import settings
from pathlib import Path

@asynccontextmanager
async def life_span(app: FastAPI):
    """Executes on application startup and shutdown."""
    print(f"[{settings.PROJECT_NAME}] Bootstrapping infrastructure...")
    
    # Absolute root discovery path calculation
    base_dir = Path(__file__).resolve().parent.parent.parent
    model_path = base_dir / "artifacts" / "iris_classifier_v1.pkl"
    
    # Core Architecture Fix: Save directly to the framework's native app.state manager
    if model_path.exists():
        app.state.ml_model = joblib.load(str(model_path))
        print(f"[{settings.PROJECT_NAME}] Production ML Model mapped safely to app.state.")
    else:
        print(f"[{settings.PROJECT_NAME}] CRITICAL: Model binary missing at {model_path}!")
        app.state.ml_model = None
        
    yield  # Serving requests window
    
    print(f"[{settings.PROJECT_NAME}] Tearing down application resources...")
    if hasattr(app.state, "ml_model"):
        del app.state.ml_model

def create_application() -> FastAPI:

    application = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.MODEL_VERSION,
        lifespan=life_span
    );

    # Register Routers (Equivalent to app.MapControllers() in .NET)
    from src.app.api.v1.predict import router as predict_router

    application.include_router(predict_router,prefix=settings.API_V1_STR,tags=["Inference Engine"]);

    return application


app = create_application();


