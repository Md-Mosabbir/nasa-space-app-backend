from utils.power_api import fetch_power_api
import pandas as pd
import numpy as np
from datetime import datetime
from utils.activities import list_activities


PARAMETERS = ['T2M', 'PRECTOT', 'WS10M', 'RH2M']  # temperature, precipitation, wind, humidity
HISTORICAL_YEARS = 10  


def fetch_origin_climate(lat, lon, start_date, end_date):
    """
    Fetch origin climate data for the past N years.
    Handles missing NASA POWER values (-999.0/-999.9) appropriately.
    
    Args:
        lat (float): latitude
        lon (float): longitude
        start_date (str): YYYYMMDD
        end_date (str): YYYYMMDD
    
    Returns:
        pd.DataFrame: historical data indexed by date
    """
    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")

    all_years = []
    current_year = datetime.now().year

    for i in range(1, HISTORICAL_YEARS + 1):
        year = current_year - i
        shifted_start = start_dt.replace(year=year)
        shifted_end = end_dt.replace(year=year)

        raw = fetch_power_api(
            temporal='daily',
            start=shifted_start.strftime("%Y%m%d"),
            end=shifted_end.strftime("%Y%m%d"),
            latitude=lat,
            longitude=lon,
            parameters=PARAMETERS
        )

        param_block = raw.get('properties', {}).get('parameter', {})
        if not param_block:
            continue

        df = pd.DataFrame(param_block)
        df.index = pd.to_datetime(df.index, format="%Y%m%d")

        # ✅ Handle missing data
        df.replace([-999.0, -999.9], np.nan, inplace=True)
        # Precipitation: treat missing as 0
        if 'PRECTOT' in df.columns:
            df['PRECTOT'] = df['PRECTOT'].fillna(0)

        all_years.append(df)

    if all_years:
        historical_df = pd.concat(all_years)
    else:
        historical_df = pd.DataFrame()

    historical_df.attrs['start_month_day'] = (start_dt.month, start_dt.day)
    historical_df.attrs['end_month_day'] = (end_dt.month, end_dt.day)

    return historical_df


def fetch_destination_historical(lat, lon, start_date, end_date, num_years=HISTORICAL_YEARS):
    """
    Fetch destination climate data for the same month/day window across past N years.
    Handles missing NASA POWER values (-999.0/-999.9) appropriately.
    
    Args:
        lat (float): latitude
        lon (float): longitude
        start_date (str): YYYYMMDD
        end_date (str): YYYYMMDD
        num_years (int): number of years to go back
    
    Returns:
        pd.DataFrame: concatenated historical data
    """
    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    all_years = []
    current_year = datetime.now().year

    for i in range(1, num_years + 1):
        year = current_year - i
        shifted_start = start_dt.replace(year=year)
        shifted_end = end_dt.replace(year=year)

        raw = fetch_power_api(
            temporal='daily',
            start=shifted_start.strftime("%Y%m%d"),
            end=shifted_end.strftime("%Y%m%d"),
            latitude=lat,
            longitude=lon,
            parameters=PARAMETERS
        )

        param_block = raw.get('properties', {}).get('parameter', {})
        if not param_block:
            continue

        df = pd.DataFrame(param_block)
        df.index = pd.to_datetime(df.index, format="%Y%m%d")

        # ✅ Handle missing data
        df.replace([-999.0, -999.9], np.nan, inplace=True)
        # Precipitation: treat missing as 0
        if 'PRECTOT' in df.columns:
            df['PRECTOT'] = df['PRECTOT'].fillna(0)

        all_years.append(df)

    if all_years:
        historical_df = pd.concat(all_years)
    else:
        historical_df = pd.DataFrame()

    historical_df.attrs['start_month_day'] = (start_dt.month, start_dt.day)
    historical_df.attrs['end_month_day'] = (end_dt.month, end_dt.day)

    return historical_df


# ----------------- Climate Statistics -----------------
def compute_statistics(dest_df, origin_stats=None):
    """
    Compute statistical summaries for the destination climate using MEDIAN 
    instead of mean (robust against outliers).
    Assumes dest_df already contains past 10 years of data from NASA POWER API.
    """
    stats = {}

    for param in ['T2M', 'RH2M', 'WS10M', 'PRECTOTCORR']:
        stats[f"{param}_median"] = dest_df[param].median()
        stats[f"{param}_std"] = dest_df[param].std()
        stats[f"{param}_p90"] = np.percentile(dest_df[param], 90)
        stats[f"{param}_p10"] = np.percentile(dest_df[param], 10)

    # Daily Discomfort Index (DI) with median
    DI_daily = dest_df['T2M'] - 0.55 * (1 - dest_df['RH2M']/100) * (dest_df['T2M'] - 14.5)
    stats['DI_median'] = DI_daily.median()
    stats['DI_std'] = DI_daily.std()

    # Precipitation probability
    stats['precip_prob'] = (dest_df['PRECTOTCORR'] > 0).mean()

    # Wind median
    stats['wind_median'] = dest_df['WS10M'].median()

    # Relative to origin stats (if provided)
    if origin_stats:
        stats['T2M_relative'] = stats['T2M_median'] - origin_stats.get('T2M_median', stats['T2M_median'])
        stats['RH2M_relative'] = stats['RH2M_median'] - origin_stats.get('RH2M_median', stats['RH2M_median'])
        stats['DI_relative'] = stats['DI_median'] - origin_stats.get('DI_median', stats['DI_median'])
    else:
        stats['T2M_relative'] = 0
        stats['RH2M_relative'] = 0
        stats['DI_relative'] = 0

    return stats

# ----------------- Risks -----------------
def compute_risks(destination_stats, activity_weight=None):
    """
    Compute risk scores using median-based statistics.
    """
    T_median = destination_stats.get('T2M_median', 25)
    DI_median = destination_stats.get('DI_median', 25)
    precip_prob = destination_stats.get('precip_prob', 0)
    wind_median = destination_stats.get('wind_median', 2)

    HOT_THRESHOLD = 30.0
    COLD_THRESHOLD = 18.0

    hot_risk = int(np.clip((T_median - 25) / (HOT_THRESHOLD - 25) * 100, 0, 100))
    cold_risk = int(np.clip((25 - T_median) / (25 - COLD_THRESHOLD) * 100, 0, 100))
    rain_risk = int(np.clip(precip_prob * 100, 0, 100))
    wind_risk = int(np.clip(wind_median / 15 * 100, 0, 100))

    if activity_weight is not None:
        hot_risk = int(hot_risk * (1 - activity_weight))
        cold_risk = int(cold_risk * (1 - activity_weight))
        rain_risk = int(rain_risk * (1 - activity_weight))
        wind_risk = int(wind_risk * (1 - activity_weight))

    return {
        "hot_risk": hot_risk,
        "cold_risk": cold_risk,
        "rain_risk": rain_risk,
        "wind_risk": wind_risk
    }

# ----------------- Adaptation Penalty -----------------
def compute_adaptation_penalty(origin_stats, dest_stats, activity=None):
    """
    Compute dynamic adaptation penalty based on user's origin vs destination.
    """
    if not origin_stats:
        return 0.2

    DI_diff = abs(dest_stats.get('DI_median', 25) - origin_stats.get('DI_median', 25))
    T_diff = abs(dest_stats.get('T2M_median', 25) - origin_stats.get('T2M_median', 25))
    RH_diff = abs(dest_stats.get('RH2M_median', 50) - origin_stats.get('RH2M_median', 50))

    activity_sensitivity = {
        'fishing': 1.0,
        'hiking': 0.8,
        'festival': 0.6,
        'generic': 0.7
    }
    factor = activity_sensitivity.get(activity, 0.7)

    penalty = factor * (0.4 * DI_diff / 10 + 0.3 * T_diff / 10 + 0.3 * RH_diff / 20)
    return float(np.clip(penalty, 0, 1))

# ----------------- Graph Generation -----------------
def generate_graph_data(origin_df, dest_df):
    def df_to_chart(df):
        if df.empty:
            return {}
        return {
            "dates": df.index.strftime("%Y-%m-%d").tolist(),
            "T2M": df['T2M'].tolist() if 'T2M' in df.columns else [],
            "RH2M": df['RH2M'].tolist() if 'RH2M' in df.columns else [],
            "WS10M": df['WS10M'].tolist() if 'WS10M' in df.columns else [],
            "PRECTOT": df['PRECTOTCORR'].tolist() if 'PRECTOTCORR' in df.columns else [],
            "DI": (df['T2M'] - 0.55 * (1 - df['RH2M']/100) * (df['T2M'] - 14.5)).tolist()
                  if 'T2M' in df.columns and 'RH2M' in df.columns else []
        }
    return {"origin": df_to_chart(origin_df), "destination": df_to_chart(dest_df)}

# ----------------- Activity Weights -----------------
def activity_weights(destination_stats, activity):
    activities_meta = list_activities()
    baseline = activities_meta.get(activity, activities_meta.get('generic', {}))
    di_min, di_max = baseline.get('DI_preferred', (16, 28))

    di = destination_stats.get('DI_median', 25)

    if di < di_min:
        weight = max(0, 1 - (di_min - di) / 10)
    elif di > di_max:
        weight = max(0, 1 - (di - di_max) / 10)
    else:
        weight = 1.0

    precip_prob = destination_stats.get('precip_prob', 0)
    weight *= (1 - 0.5 * precip_prob)

    return float(np.clip(weight, 0, 1))

# ----------------- Comfort Index -----------------
def compute_comfort_index(destination_stats, activity_weight, adaptation_penalty):
    DI = destination_stats.get('DI_median', 25)
    precip_prob = destination_stats.get('precip_prob', 0)
    wind_median = destination_stats.get('wind_median', 2)

    DI_score = np.clip((DI - 15) / (30 - 15), 0, 1)
    precip_score = 1 - np.clip(precip_prob, 0, 1)
    wind_score = 1 - np.clip(wind_median / 10, 0, 1)

    combined_score = (0.5 * DI_score + 0.3 * precip_score + 0.2 * wind_score)
    weighted_score = combined_score * activity_weight
    final_score = weighted_score * (1 - adaptation_penalty)

    final_score_100 = int(np.clip(final_score * 100, 0, 100))

    if final_score_100 >= 80:
        category, color = "Excellent", "green"
    elif final_score_100 >= 60:
        category, color = "Good", "yellowgreen"
    elif final_score_100 >= 40:
        category, color = "Moderate", "yellow"
    elif final_score_100 >= 20:
        category, color = "Poor", "orange"
    else:
        category, color = "Very Poor", "red"

    main_factors = []
    if DI_score < 0.5:
        main_factors.append('Temperature/DI')
    if precip_score < 0.5:
        main_factors.append('Precipitation')
    if wind_score < 0.5:
        main_factors.append('Wind')

    return {
        'score': final_score_100,
        'category': category,
        'color': color,
        'main_factors': main_factors
    }


def analysis(origin_lat, origin_lon, dest_lat, dest_lon, start_date, end_date, activities=None, include_graphs=True):
    """
    Run full climate + activity comfort analysis and return structured JSON.
    Optionally include daily data for charting (Chart.js, etc.) without extra fetches.

    Args:
        origin_lat, origin_lon: User's origin coordinates
        dest_lat, dest_lon: Destination coordinates
        start_date, end_date: YYYYMMDD
        activities (list): List of activity strings (optional)
        include_graphs (bool): Whether to include daily data for charts

    Returns:
        dict: Structured JSON including origin, destination, activities, meta, and optional graphs
    """

    def convert_numpy(obj):
        """Recursively convert NumPy types to Python native types for JSON."""
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(x) for x in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        else:
            return obj

    def compute_dynamic_adaptation_penalty(origin_stats, dest_stats, activity=None):
        if not origin_stats:
            return 0.2  # fallback default

        DI_diff = abs(dest_stats.get('DI_mean', 25) - origin_stats.get('DI_mean', 25))
        T_diff = abs(dest_stats.get('T2M_mean', 25) - origin_stats.get('T2M_mean', 25))
        RH_diff = abs(dest_stats.get('RH2M_mean', 50) - origin_stats.get('RH2M_mean', 50))

        activity_sensitivity = {
            'fishing': 1.0,
            'hiking': 0.8,
            'festival': 0.6,
            'generic': 0.7
        }
        factor = activity_sensitivity.get(activity, 0.7)
        penalty = factor * (0.4 * DI_diff / 10 + 0.3 * T_diff / 10 + 0.3 * RH_diff / 20)
        return float(np.clip(penalty, 0, 1))

    if activities is None:
        activities = list(list_activities().keys())

    # 1️⃣ Fetch climate data
    origin_df = fetch_origin_climate(origin_lat, origin_lon, start_date, end_date)
    dest_df = fetch_destination_historical(dest_lat, dest_lon, start_date, end_date)

    # 2️⃣ Compute statistics
    origin_stats = compute_statistics(origin_df) if not origin_df.empty else None
    dest_stats = compute_statistics(dest_df, origin_stats=origin_stats)

    # 3️⃣ Compute activity weights, comfort indices, and per-activity risks
    activities_dict = {}
    for activity in activities:
        weight = activity_weights(dest_stats, activity)
        adaptation_penalty = compute_dynamic_adaptation_penalty(origin_stats, dest_stats, activity)
        comfort_index = compute_comfort_index(dest_stats, weight, adaptation_penalty)
        activity_risks = compute_risks(dest_stats, activity_weight=weight)
        activities_dict[activity] = {
            "weight": weight,
            "adaptation_penalty": adaptation_penalty,
            "comfort_index": comfort_index,
            "risks": activity_risks
        }

    # 4️⃣ Build structured JSON
    result_json = {
        "origin": {
            "latitude": origin_lat,
            "longitude": origin_lon,
            "statistics": origin_stats
        },
        "destination": {
            "latitude": dest_lat,
            "longitude": dest_lon,
            "statistics": dest_stats,
        },
        "activities": activities_dict,
        "meta": {
            "start_date": start_date,
            "end_date": end_date,
            "historical_years": HISTORICAL_YEARS,
            "notes": "DI calculated using NASA POWER data; missing precipitation treated as 0; adaptation penalty dynamic"
        }
    }

    # 5️⃣ Optionally add chart/graph data
    if include_graphs:
        result_json['graphs'] = generate_graph_data(origin_df, dest_df)

    # ✅ Convert all NumPy types to native Python types
    return convert_numpy(result_json)

# ----------------- Multi-Scenario Test Block -----------------
if __name__ == "__main__":
    test_scenarios = {
        "Normal (Dhaka → Sri Lanka)": {"origin": (23.8103, 90.4125), "dest": (7.8731, 80.7718)},  # Sri Lanka
        "Moderate Cold (Dhaka → New York, USA)": {"origin": (23.8103, 90.4125), "dest": (40.7128, -74.0060)},  # New York
        "Hot Extreme (Dhaka → Sahara)": {"origin": (23.8103, 90.4125), "dest": (23.4162, 25.6628)},  # Sahara Desert
        "Cold Extreme (Dhaka → Antarctica)": {"origin": (23.8103, 90.4125), "dest": (-82.8628, 135.0000)},  # Antarctica
        "Rainy/Windy (Dhaka → Scotland)": {"origin": (23.8103, 90.4125), "dest": (56.4907, -4.2026)},  # Scotland
        "Mild (Dhaka → Japan)": {"origin": (23.8103, 90.4125), "dest": (36.2048, 138.2529)},  # Japan
    }

    start_date = "20251001"
    end_date = "20251005"
    activities = ['fishing', 'hiking', 'festival', 'generic']

    import pprint

    for scenario_name, coords in test_scenarios.items():
        print(f"=== Scenario: {scenario_name} ===")
        output = analysis(
            origin_lat=coords["origin"][0], origin_lon=coords["origin"][1],
            dest_lat=coords["dest"][0], dest_lon=coords["dest"][1],
            start_date=start_date,
            end_date=end_date,
            activities=activities
        )
        pprint.pprint(output)
        print("\n")
