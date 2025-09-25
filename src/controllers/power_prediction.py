from utils.power_api import fetch_power_api
import pandas as pd
import numpy as np
from datetime import datetime


# ---------------------- MAIN ANALYSIS ---------------------- #
def analysis(data):
    # Step 1: fetch NASA POWER data (5 years)
    raw = fetch(data["location"]["lat"], data["location"]["lon"])

    # Step 2: clean dataframe
    df = pandify(raw['properties']['parameter'])

    # Step 3: scope dataframe to requested range
    df_scoped = scope(df, data["date_range"]["start"], data["date_range"]["end"])

    # Step 4: calculate probabilities and colors per activity
    comfort_risk = {}
    for activity in data["activities"]:
        probs = probabilities(df_scoped, activity)
        colors = color_coding(probs)
        comfort_risk[activity] = {"probabilities": probs, "color_codes": colors}

    return {
        "date_range": {
            "start": data["date_range"]["start"],
            "end": data["date_range"]["end"],
        },
        "location": {
            "lat": data["location"]["lat"],
            "lon": data["location"]["lon"],
        },
        "activities": data["activities"],
        "comfort_risk": comfort_risk,
        "guidance": "Green: Good to go! Yellow: Caution. Red: Consider plan B.",
    }


# ---------------------- HELPERS ---------------------- #
def fetch(lat, lon):
    return fetch_power_api(
        "daily",
        2020,
        2025,
        lat,
        lon,
        ["T2M", "PRECTOT", "WS10M", "RH2M"],
    )


def pandify(data: dict) -> pd.DataFrame:
    df = pd.DataFrame(data)
    df = df.replace(-999.0, np.nan)
    df = df.dropna()
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    return df


def scope(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")

    start_md = (start_dt.month, start_dt.day)
    end_md = (end_dt.month, end_dt.day)

    mask = (
        (df.index.month > start_md[0])
        | ((df.index.month == start_md[0]) & (df.index.day >= start_md[1]))
    ) & (
        (df.index.month < end_md[0])
        | ((df.index.month == end_md[0]) & (df.index.day <= end_md[1]))
    )

    return df.loc[mask]


# ---------------------- ACTIVITY-SPECIFIC PROBABILITIES ---------------------- #
def probabilities(df: pd.DataFrame, activity: str) -> dict:
    thresholds = {
        "hiking": {"very_hot": 35, "very_cold": 5, "very_windy": 10, "very_wet": 5, "very_uncomfortable": 80},
        "paragliding": {"very_hot": 30, "very_cold": 0, "very_windy": 6, "very_wet": 2, "very_uncomfortable": 85},
        "fishing": {"very_hot": 32, "very_cold": 2, "very_windy": 8, "very_wet": 10, "very_uncomfortable": 90},
    }

    th = thresholds.get(activity, thresholds["hiking"])  # default to hiking if unknown
    total_days = len(df)
    if total_days == 0:
        return {k: None for k in th}

    return {
        "very_hot": round((df["T2M"] > th["very_hot"]).sum() / total_days, 2),
        "very_cold": round((df["T2M"] < th["very_cold"]).sum() / total_days, 2),
        "very_windy": round((df["WS10M"] > th["very_windy"]).sum() / total_days, 2),
        "very_wet": round((df["PRECTOTCORR"] > th["very_wet"]).sum() / total_days, 2),
        "very_uncomfortable": round((df["RH2M"] > th["very_uncomfortable"]).sum() / total_days, 2),
    }


def color_coding(probs: dict) -> dict:
    colors = {}
    for k, v in probs.items():
        if v is None:
            colors[k] = "gray"
        elif v < 0.3:
            colors[k] = "green"
        elif v < 0.6:
            colors[k] = "yellow"
        else:
            colors[k] = "red"
    return colors
