from fastapi import FastAPI
from routes import analyse
app = FastAPI(
    title="NASA Space App Backend",
    description="API for NASA Space App backend services.",
    version="1.0.0",
    contact={
        "name": "Mosabbir",
        "email": "kmosabbir@gmail.com",
    },
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {
            "name": "Health",
            "description": "Endpoints for health checks."
        }
    ]
)

@app.get("/", summary="Root Endpoint")
def read_root():
    return {"message": "Welcome to the NASA Space App Backend!"}

@app.get("/health", summary="Health Check", tags=["Health"])
def health_check():
    return {"status": "ok"}

# Include the router from routes/analyse.py
app.include_router(analyse.router)
