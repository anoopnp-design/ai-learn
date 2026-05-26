from fastapi import FastAPI
from src.app.core.config import settings
from contextlib import asynccontextmanager

app_state ={}; # Dependency container 

@asynccontextmanager
async def life_span(app:FastAPI):
   # STARTUP: Initialize/Load resource-heavy assets here
    print(f"[{settings.PROJECT_NAME}] Bootstrapping infrastructure...")
    
    # Simulated Model Loading (We will place the real model binary state here later)
    app_state["ml_model"] = {"model_name": settings.MODEL_NAME, "status": "active"}
    print(f"[{settings.PROJECT_NAME}] ML Model loaded successfully into memory state.")
    
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


