from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from routes import analyse
from routes import ai

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

# ✅ CORS Middleware
origins = [
    "http://localhost:5173",  # Vite dev server
    "http://localhost:3000",  # other ports if needed
    "*",  # for testing only, allow all
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", summary="Root Endpoint")
def read_root():
    return {"message": "Welcome to the NASA Space App Backend!"}

@app.get("/health", summary="Health Check", tags=["Health"])
def health_check():
    return {"status": "ok"}

# Include the router from routes/analyse.py
app.include_router(analyse.router)
app.include_router(ai.router)
