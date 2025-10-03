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


def compute_statistics(dest_df, origin_stats=None):
    """
    Compute statistical summaries for the destination climate using a proven DI formula.
    Optionally compare to origin climate for relative discomfort.

    Args:
        dest_df (pd.DataFrame): Historical climate data (scoped to date range)
            Columns: ['T2M', 'RH2M', 'WS10M', 'PRECTOTCORR']
        origin_stats (dict, optional): Precomputed statistics from user's origin.
            Example: {'T2M_mean': 22.0, 'RH2M_mean': 60.0}

    Returns:
        dict: Statistical summary
    """

    stats = {}


    for param in ['T2M', 'RH2M', 'WS10M', 'PRECTOTCORR']:
        stats[f"{param}_mean"] = dest_df[param].mean()
        stats[f"{param}_std"] = dest_df[param].std()
        stats[f"{param}_p90"] = np.percentile(dest_df[param], 90)
        stats[f"{param}_p10"] = np.percentile(dest_df[param], 10)


    # DI = T - 0.55 * (1 - RH/100) * (T - 14.5)
    DI_daily = dest_df['T2M'] - 0.55 * (1 - dest_df['RH2M']/100) * (dest_df['T2M'] - 14.5)
    stats['DI_mean'] = DI_daily.mean()
    stats['DI_std'] = DI_daily.std()


    stats['precip_prob'] = (dest_df['PRECTOTCORR'] > 0).mean()


    stats['wind_mean'] = dest_df['WS10M'].mean()


    if origin_stats:
        stats['T2M_relative'] = stats['T2M_mean'] - origin_stats.get('T2M_mean', stats['T2M_mean'])
        stats['RH2M_relative'] = stats['RH2M_mean'] - origin_stats.get('RH2M_mean', stats['RH2M_mean'])
        stats['DI_relative'] = stats['DI_mean'] - origin_stats.get('DI_mean', stats['DI_mean'])
    else:
        stats['T2M_relative'] = 0
        stats['RH2M_relative'] = 0
        stats['DI_relative'] = 0

    return stats

def compute_risks(destination_stats, activity_weight=None):
    """
    Compute risk scores for hot, cold, rain, and wind.
    If activity_weight is provided, risks are adjusted per activity.

    Args:
        destination_stats (dict): Output from compute_statistics
        activity_weight (float, optional): activity-specific weight [0,1]

    Returns:
        dict: risk scores (0–100)
    """
    T_mean = destination_stats.get('T2M_mean', 25)
    DI_mean = destination_stats.get('DI_mean', 25)
    precip_prob = destination_stats.get('precip_prob', 0)
    wind_mean = destination_stats.get('wind_mean', 2)

    # Hot/cold thresholds
    HOT_THRESHOLD = 30.0
    COLD_THRESHOLD = 18.0

    hot_risk = int(np.clip((T_mean - 25) / (HOT_THRESHOLD - 25) * 100, 0, 100))
    cold_risk = int(np.clip((25 - T_mean) / (25 - COLD_THRESHOLD) * 100, 0, 100))
    rain_risk = int(np.clip(precip_prob * 100, 0, 100))
    wind_risk = int(np.clip(wind_mean / 15 * 100, 0, 100))

    # Adjust by activity weight if provided (lower weight = higher perceived risk)
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


def compute_adaptation_penalty(origin_stats, dest_stats, activity=None):
    """
    Compute dynamic adaptation penalty based on user's origin vs destination.
    
    Args:
        origin_stats (dict): Origin climate statistics
        dest_stats (dict): Destination climate statistics
        activity (str, optional): Activity type, can tweak sensitivity
    
    Returns:
        float: Adaptation penalty [0,1]
    """
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
    
    # Clip to 0–1
    return float(np.clip(penalty, 0, 1))

# ----------------- Graph Generation -----------------
def generate_graph_data(origin_df, dest_df):
    """
    Generate structured data for charting without losing any raw info.
    Returns daily values for key parameters.
    """
    def df_to_chart(df):
        if df.empty:
            return {}
        chart_data = {
            "dates": df.index.strftime("%Y-%m-%d").tolist(),
            "T2M": df['T2M'].tolist() if 'T2M' in df.columns else [],
            "RH2M": df['RH2M'].tolist() if 'RH2M' in df.columns else [],
            "WS10M": df['WS10M'].tolist() if 'WS10M' in df.columns else [],

            "PRECTOT": df['PRECTOTCORR'].tolist() if 'PRECTOTCORR' in df.columns else [],
            # Optional: Discomfort Index per day
            "DI": (df['T2M'] - 0.55 * (1 - df['RH2M']/100) * (df['T2M'] - 14.5)).tolist()
                  if 'T2M' in df.columns and 'RH2M' in df.columns else []
        }
        return chart_data

    return {
        "origin": df_to_chart(origin_df),
        "destination": df_to_chart(dest_df)
    }

# ----------------- Activity Weight Function -----------------
def activity_weights(destination_stats, activity):
    """
    Compute activity weight based on destination climate.
    Higher weight = more comfortable for the activity.
    
    Args:
        destination_stats (dict): Output from compute_statistics
        activity (str): Activity type (e.g., 'fishing', 'hiking', 'festival')
    
    Returns:
        float: Activity weight [0, 1]
    """
    # Pull activity DI preferences from utils.activities
    activities_meta = list_activities()
    baseline = activities_meta.get(activity, activities_meta.get('generic', {}))
    di_min, di_max = baseline.get('DI_preferred', (16, 28))

    di = destination_stats.get('DI_mean', 25)

    # Weight based on DI within preferred range (linear scaling)
    if di < di_min:
        weight = max(0, 1 - (di_min - di) / 10)  # colder than optimal
    elif di > di_max:
        weight = max(0, 1 - (di - di_max) / 10)  # hotter than optimal
    else:
        weight = 1.0  # optimal range


    precip_prob = destination_stats.get('precip_prob', 0)
    weight *= (1 - 0.5 * precip_prob)  # reduce weight if raining

    # Clip to 0-1
    return float(np.clip(weight, 0, 1))



def compute_comfort_index(destination_stats, activity_weight, adaptation_penalty):
    """
    Compute overall comfort index for an activity at a destination.
    
    Args:
        destination_stats (dict): Output from compute_statistics
        activity_weight (float): Weight of the activity [0, 1]
        adaptation_penalty (float): Penalty from adaptation [0, 1]
    
    Returns:
        dict: {'score': int, 'category': str, 'color': str, 'main_factors': list}
    """

    # Extract key stats
    DI = destination_stats.get('DI_mean', 25)
    precip_prob = destination_stats.get('precip_prob', 0)
    wind_mean = destination_stats.get('wind_mean', 2)


    # DI subscore: 15–30 optimal range
    DI_score = np.clip((DI - 15) / (30 - 15), 0, 1)

    # Precipitation subscore: less rain better
    precip_score = 1 - np.clip(precip_prob, 0, 1)

    # Wind subscore: calmer better, assume 0–10 m/s
    wind_score = 1 - np.clip(wind_mean / 10, 0, 1)

    # Combine normalized scores
    combined_score = (0.5 * DI_score + 0.3 * precip_score + 0.2 * wind_score)

    # Apply activity weight
    weighted_score = combined_score * activity_weight

    # Apply adaptation penalty
    final_score = weighted_score * (1 - adaptation_penalty)

    # Scale to 0–100
    final_score_100 = int(np.clip(final_score * 100, 0, 100))

    # Determine category
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

    # Identify main limiting factors
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


    origin_df = fetch_origin_climate(origin_lat, origin_lon, start_date, end_date)
    dest_df = fetch_destination_historical(dest_lat, dest_lon, start_date, end_date)


    origin_stats = compute_statistics(origin_df) if not origin_df.empty else None
    dest_stats = compute_statistics(dest_df, origin_stats=origin_stats)


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


    if include_graphs:
        result_json['graphs'] = generate_graph_data(origin_df, dest_df)


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
