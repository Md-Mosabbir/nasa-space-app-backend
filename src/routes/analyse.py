
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from utils.power_api import fetch_power_api
from controllers.power_prediction import run_analysis

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
    
    # extract info from request
    lat = data["location"]["lat"]
    lon = data["location"]["lon"]
    start_date = data["date_range"]["start"]
    end_date = data["date_range"]["end"]
    activities = data["activities"]
    
    results = {}
    for activity in activities:
        results[activity] = run_analysis(
            activity=activity,
            lat=lat,
            lon=lon,
            start_date=start_date,
            end_date=end_date,
            make_plots=False
        )
    
    return results

    