
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
    name: str | None = None  # optional place name
    lat: float
    lon: float

class ActivityRequest(BaseModel):
    date_range: DateRange
    activities: List[str]
    origin: Location
    destination: Location


@router.post("/activity", summary="Analyse Activity Weather")
def analyse_activity_weather(request: ActivityRequest):
    data = request.model_dump()
    return analysis(
        origin_lat=data['origin']['lat'],
        origin_lon=data['origin']['lon'],
        dest_lat=data['destination']['lat'],
        dest_lon=data['destination']['lon'],
        start_date=data['date_range']['start'],
        end_date=data['date_range']['end'],
        activities=data.get('activities')
    )
