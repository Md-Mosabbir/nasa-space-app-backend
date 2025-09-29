"""
nasa_event_risk.py

Fetch NASA POWER (daily/hourly) time series for a user-specified location and date range,
then compute historical-likelihood style analyses for outdoor activity risk.

- If start_date == end_date -> hourly endpoint is used.
- Otherwise -> daily endpoint is used.

Dependencies:
  pip install pandas numpy requests matplotlib
"""

from typing import Optional, Dict, Any, List, Tuple
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import json
import csv
import os
import matplotlib.pyplot as plt

# ------------------------
# Configuration / metadata
# ------------------------
NASA_VARIABLES = {
    "temp": {"code": "T2M", "unit": "C"},
    "precipitation": {"code": "PRECTOT", "unit": "mm"},
    "wind speed": {"code": "WS10M", "unit": "m/s"},
    "humidity": {"code": "RH2M", "unit": "%"},
}

DEFAULT_THRESHOLDS = {
    "hiking": {"temp": [10, 25], "wind speed": [0, 20], "humidity": [30, 60], "precipitation": [0, 2]},
    "paragliding": {"temp": [15, 28], "wind speed": [10, 25], "humidity": [30, 65], "precipitation": [0, 1]},
    "fishing": {"temp": [10, 30], "wind speed": [0, 25], "humidity": [30, 80], "precipitation": [0, 3]},
}

# Color thresholds for comfort percentage
COLOR_THRESHOLDS = [
    (100, "green", "Comfortable"),
    (75, "yellow", "Safe"),
    (50, "orange", "Moderate"),
    (0, "red", "Risky"),
]


# ------------------------
# Small helpers
# ------------------------
def choose_color_and_suggestion(pct: float) -> Tuple[str, str]:
    for thr, color, suggestion in COLOR_THRESHOLDS:
        if pct >= thr:
            return color, suggestion
    return "red", "Risky"


def make_bar(pct: float, length: int = 20) -> str:
    filled = int((pct / 100) * length)
    filled = max(0, min(filled, length))
    return "█" * filled + "░" * (length - filled) + f" {pct:.1f}%"


# ------------------------
# Discomfort flags helpers
# ------------------------
def discomfort_flags_from_values(values: Dict[str, float], thresholds: Dict[str, List[float]]) -> List[str]:
    flags = []
    if "temp" in values:
        if values["temp"] < thresholds["temp"][0]:
            flags.append("very cold")
        elif values["temp"] > thresholds["temp"][1]:
            flags.append("very hot")
    if "wind speed" in values:
        if values["wind speed"] > thresholds["wind speed"][1]:
            flags.append("very windy")
    if "precipitation" in values:
        if values["precipitation"] > thresholds["precipitation"][1]:
            flags.append("very wet")
    if "humidity" in values:
        if values["humidity"] < thresholds["humidity"][0] or values["humidity"] > thresholds["humidity"][1]:
            flags.append("very uncomfortable")
    return flags or ["good conditions"]


def suggestion_text_from_flags(flags: List[str]) -> str:
    if flags == ["good conditions"]:
        return "Conditions look favorable for your activity."
    return "Conditions may be: " + ", ".join(flags)


# ------------------------
# NASA POWER API fetcher (hourly if single-day, daily otherwise)
# ------------------------
def fetch_power_data(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch POWER data from NASA:
    - Uses the hourly endpoint if start_date == end_date
    - Otherwise uses the daily endpoint.

    Returns a DataFrame indexed by datetime with columns renamed to friendly names:
    'temp', 'precipitation', 'wind speed', 'humidity' (only those present in response).
    """
    # validate dates
    try:
        sd = datetime.strptime(start_date, "%Y-%m-%d")
        ed = datetime.strptime(end_date, "%Y-%m-%d")
    except Exception as e:
        raise ValueError("start_date and end_date must be 'YYYY-MM-DD'") from e

    # choose endpoint
    if sd.date() == ed.date():
        # single-day -> hourly
        endpoint = "hourly"
    else:
        endpoint = "daily"

    base = "https://power.larc.nasa.gov/api/temporal"
    params_codes = ",".join([v["code"] for v in NASA_VARIABLES.values()])
    start_str = start_date.replace("-", "")
    end_str = end_date.replace("-", "")
    url = f"{base}/{endpoint}/point?start={start_str}&end={end_str}&latitude={lat}&longitude={lon}&parameters={params_codes}&format=JSON&community=RE"

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    params = data.get("properties", {}).get("parameter", {})
    if not params:
        return pd.DataFrame()

    # build DataFrame from parameter dicts: keys are codes -> dict(date->value)
    # Convert to DataFrame by aligning on date keys
    df = pd.DataFrame({code: pd.Series(vals) for code, vals in params.items()})
    # index are date strings: try to parse intelligently
    # daily typically 'YYYYMMDD', hourly often 'YYYYMMDDHH' (or with min/sec)
    try:
        sample_idx = next(iter(df.index))
        if len(sample_idx) == 8:  # YYYYMMDD
            df.index = pd.to_datetime(df.index, format="%Y%m%d")
        else:
            # Let pandas infer (works for many formats, including hourly-ish)
            df.index = pd.to_datetime(df.index, infer_datetime_format=True, errors="coerce")
    except Exception:
        df.index = pd.to_datetime(df.index, infer_datetime_format=True, errors="coerce")

    # Map parameter codes to friendly keys (if present)
    codes_to_keys = {v["code"]: k for k, v in NASA_VARIABLES.items()}
    col_map = {col: codes_to_keys[col] for col in df.columns if col in codes_to_keys}
    df = df.rename(columns=col_map)

    # Keep only our known variables (in case API returned extras)
    df = df[[k for k in NASA_VARIABLES.keys() if k in df.columns]].copy()

    # Clean sentinel / missing values
    df = df.replace(-999.0, np.nan).dropna(how="any")

    # If the response index has timezone-naive hourly timestamps, no timezone conversion is done.
    return df


# ------------------------
# Exceedance & trend utilities
# ------------------------
def exceedance_probability(series: pd.Series, threshold: float, direction: str = "above") -> float:
    if series.empty:
        return float("nan")
    if direction == "above":
        return float((series > threshold).sum()) / len(series)
    return float((series < threshold).sum()) / len(series)


def long_term_trend(series: pd.Series) -> Dict[str, Any]:
    if series.empty:
        return {"slope": None, "intercept": None, "r2": None, "n": 0}
    yearly = series.groupby(series.index.year).mean().dropna()
    if len(yearly) < 2:
        return {"slope": 0.0, "intercept": float(yearly.iloc[0]) if not yearly.empty else None, "r2": None, "n": len(yearly)}
    x = np.array(list(yearly.index))
    y = yearly.values.astype(float)
    A = np.vander(x, 2)
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    slope = coeffs[0]
    intercept = coeffs[1]
    y_pred = slope * x + intercept
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else None
    return {"slope": float(slope), "intercept": float(intercept), "r2": float(r2) if r2 is not None else None, "n": len(yearly)}


# ------------------------
# Analysis (quartiles, IQR, ranking, mean, flags, exceedances, trends)
# ------------------------
def analyze_period(df_period: pd.DataFrame, activity: str, thresholds: Dict[str, Dict[str, List[float]]],
                   params: List[str]) -> Dict[str, Any]:
    th = thresholds.get(activity, thresholds.get("hiking"))
    params_present = [p for p in params if p in df_period.columns]
    if not params_present:
        return {"error": "No requested variables present in data"}

    # Quartiles (per-variable)
    quartiles = {}
    for p in params_present:
        s = df_period[p]
        quartiles[p] = {"Q1": float(s.quantile(0.25)), "Q3": float(s.quantile(0.75))}

    # Filter to rows where all variables are within their per-variable IQR (intersection)
    mask = pd.Series(True, index=df_period.index)
    for p in params_present:
        q1, q3 = quartiles[p]["Q1"], quartiles[p]["Q3"]
        mask &= (df_period[p] >= q1) & (df_period[p] <= q3)
    df_iqr = df_period.loc[mask]

    # Score each IQR row: +1 within activity threshold, -1 outside
    scores = {}
    n = len(params_present)
    min_possible = -n
    max_possible = n
    for ts, row in df_iqr.iterrows():
        raw_score = 0
        values = {}
        for p in params_present:
            min_val, max_val = th[p]
            v = float(row[p])
            values[p] = v
            raw_score += 1 if (min_val <= v <= max_val) else -1
        pct = ((raw_score - min_possible) / (max_possible - min_possible)) * 100
        pct = round(pct, 1)
        color, _ = choose_color_and_suggestion(pct)
        flags = discomfort_flags_from_values(values, th)
        suggestion = suggestion_text_from_flags(flags)
        scores[str(ts)] = {
            "values": values,
            "raw_score": int(raw_score),
            "comfort_score_pct": pct,
            "score_bar": make_bar(pct),
            "color": color,
            "flags": flags,
            "suggestion": suggestion,
        }

    ranked = dict(sorted(scores.items(), key=lambda kv: kv[1]["comfort_score_pct"], reverse=True))
    best = dict(list(ranked.items())[:2])
    worst = dict(list(ranked.items())[-2:])

    # Mean analysis across the entire requested period (not restricted to IQR)
    means = {p: float(df_period[p].mean()) for p in params_present}
    raw_score_mean = 0
    for p in params_present:
        min_val, max_val = th[p]
        val = means[p]
        raw_score_mean += 1 if (min_val <= val <= max_val) else -1
    pct_mean = ((raw_score_mean - min_possible) / (max_possible - min_possible)) * 100
    pct_mean = round(pct_mean, 1)
    color_mean, _ = choose_color_and_suggestion(pct_mean)
    flags_mean = discomfort_flags_from_values(means, th)
    suggestion_mean = suggestion_text_from_flags(flags_mean)

    # Exceedance probabilities for thresholds
    exceedances = {}
    for p in params_present:
        low, high = th[p]
        exceedances[p] = {
            "prob_below_low_pct": round(exceedance_probability(df_period[p], low, direction="below") * 100, 1),
            "prob_above_high_pct": round(exceedance_probability(df_period[p], high, direction="above") * 100, 1),
        }

    # Trends
    trends = {p: long_term_trend(df_period[p]) for p in params_present}

    return {
        "quartiles": quartiles,
        "iqr_entries_count": len(df_iqr),
        "scores": scores,
        "ranked": ranked,
        "best": best,
        "worst": worst,
        "mean_summary": {
            "means": means,
            "raw_score": int(raw_score_mean),
            "comfort_score_pct": pct_mean,
            "score_bar": make_bar(pct_mean),
            "color": color_mean,
            "flags": flags_mean,
            "suggestion": suggestion_mean,
        },
        "exceedances_percent": exceedances,
        "trends_per_variable": trends,
    }


# ------------------------
# Exports & plots (optional)
# ------------------------
def export_results_to_json(results: dict, outpath: str) -> None:
    meta = {
        "units": {k: NASA_VARIABLES[k]["unit"] for k in NASA_VARIABLES if k in results.get("mean_summary", {}).get("means", {})},
        "source": "NASA POWER (https://power.larc.nasa.gov/)",
        "generated_at": datetime.utcnow().isoformat() + "Z"
    }
    out = {"metadata": meta, "results": results}
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def plot_time_series(df_period: pd.DataFrame, params: List[str], title: Optional[str] = None, savepath: Optional[str] = None):
    n = len(params)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), squeeze=False)
    for i, p in enumerate(params):
        ax = axes[i, 0]
        ax.plot(df_period.index, df_period[p])
        ax.set_ylabel(f"{p} ({NASA_VARIABLES[p]['unit']})")
        ax.grid(True)
    if title:
        fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if savepath:
        plt.savefig(savepath)
    else:
        plt.show()
    plt.close(fig)


# ------------------------
# Top-level run
# ------------------------
def run_analysis(activity: str,
                 lat: float,
                 lon: float,
                 start_date: str,
                 end_date: str,
                 variables: Optional[List[str]] = None,
                 thresholds: Optional[Dict[str, Dict[str, List[float]]]] = None,
                 export_dir: Optional[str] = None,
                 make_plots: bool = False) -> Dict[str, Any]:
    """
    Unified entry point.

    - activity: e.g. "hiking"
    - lat, lon: floats (user-provided)
    - start_date, end_date: 'YYYY-MM-DD'
    - variables: subset of NASA_VARIABLES keys (default: all)
    - thresholds: override (default DEFAULT_THRESHOLDS)
    - export_dir: optional path to save JSON/plots
    - make_plots: whether to produce time-series plot and save (requires matplotlib)
    """
    if not (start_date and end_date):
        raise ValueError("start_date and end_date required in 'YYYY-MM-DD' format")

    params = variables or list(NASA_VARIABLES.keys())
    params = [p for p in params if p in NASA_VARIABLES]
    if not params:
        raise ValueError("No valid variables requested")

    th_map = thresholds or DEFAULT_THRESHOLDS

    # Fetch data (automatically picks hourly if single-day)
    df = fetch_power_data(lat, lon, start_date, end_date)
    if df.empty:
        return {"error": "No data returned for the requested location/date. Check the coordinates, variables availability, and POWER API limits."}

    # If some requested variables are missing in the returned df, warn in the result
    missing = [p for p in params if p not in df.columns]
    if missing:
        # we will analyze only present variables
        params_present = [p for p in params if p in df.columns]
    else:
        params_present = params

    results = analyze_period(df[params_present], activity, th_map, params_present)

    top = {
        "location": {"lat": lat, "lon": lon},
        "activity": activity,
        "period": {"start": start_date, "end": end_date},
        "variables_requested": params,
        "variables_used": params_present,
        "missing_variables": missing,
        "results": results
    }

    if export_dir:
        os.makedirs(export_dir, exist_ok=True)
        json_path = os.path.join(export_dir, f"analysis_{activity}_{start_date}_{end_date}.json")
        export_results_to_json(results, json_path)
        top["exported_json"] = json_path
        if make_plots:
            ts_path = os.path.join(export_dir, f"time_series_{activity}_{start_date}_{end_date}.png")
            plot_time_series(df[params_present], params_present, title=f"{activity} @ ({lat},{lon}) {start_date}..{end_date}", savepath=ts_path)
            top["plot_time_series"] = ts_path
    else:
        if make_plots:
            plot_time_series(df[params_present], params_present, title=f"{activity} @ ({lat},{lon}) {start_date}..{end_date}")

    return top


# ------------------------
# Example usage
# ------------------------
if __name__ == "__main__":
    # Replace these with user input values in real usage
    example_activity = "hiking"
    example_lat, example_lon = 23.8103, 90.4125  # Dhaka
    example_start, example_end = "2020-06-01", "2020-06-10"  # multi-day -> daily analysis

    result = run_analysis(
        activity=example_activity,
        lat=example_lat,
        lon=example_lon,
        start_date=example_start,
        end_date=example_end,
        variables=["temp", "precipitation", "wind speed", "humidity"],
        export_dir=None,
        make_plots=False
    )

    print(json.dumps(result, indent=2))
