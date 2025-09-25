
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from utils.power_api import fetch_power_api
from controllers.power_prediction import analysis

router = APIRouter(
    prefix="/analyse",
    tags=["Analyse"],
)

@router.get("/", summary="Analyse Endpoint")
def analyse_endpoint():
    return {"message": "Analyse endpoint is working!"}

class DateRange(BaseModel):
    start: str
    end: str

class Location(BaseModel):
    lat: float
    lon: float

class ActivityRequest(BaseModel):
    date_range: DateRange
    activities: List[str]
    location: Location


@router.post("/activity", summary="Analyse Activity Weather")
def analyse_activity_weather(request: ActivityRequest):
    data = request.model_dump()
    return analysis(data)

    