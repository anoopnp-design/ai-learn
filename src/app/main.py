import joblib
from fastapi import FastAPI
from src.app.core.config import settings
from contextlib import asynccontextmanager

app_state ={}; # Dependency container 

@asynccontextmanager
async def life_span(app:FastAPI):
   # STARTUP: Initialize/Load resource-heavy assets here
    print(f"[{settings.PROJECT_NAME}] Bootstrapping infrastructure...")
    
    # Core Architecture Fix: Load our compiled binary model state into memory once
    model_path = "artifacts/iris_classifier_v1.pkl"
    
    if joblib.os.path.exists(model_path):
        app_state["ml_model"] = joblib.load(model_path)
        print(f"[{settings.PROJECT_NAME}] Production ML Model deserialized successfully into state memory.")
    else:
        print(f"[{settings.PROJECT_NAME}] CRITICAL: Model binary artifact not found at {model_path}!")
        app_state["ml_model"] = None
    
    yield  # The application runs and processes HTTP requests here
    
    # SHUTDOWN: Clean up references, close DB pools or file streams
    print(f"[{settings.PROJECT_NAME}] Tearing down infrastructure resources...")
    app_state.clear()

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


