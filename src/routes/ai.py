from fastapi import APIRouter
from pydantic import BaseModel
from typing import Any
from controllers.ai import ask_ai

router = APIRouter(
    prefix="/ai",
    tags=["AI"]
)

class AIRequest(BaseModel):
    analyzed_data: dict  # the whole JSON from /analyse
    user_message: str | None = None  # optional follow-up

@router.post("/ask", summary="Ask AI for weather advice")
def ask_ai_endpoint(request: AIRequest):
    return {"reply": ask_ai(request.analyzed_data, request.user_message)}
