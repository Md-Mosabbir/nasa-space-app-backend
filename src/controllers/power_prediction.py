from ..utils.power_api import fetch_power_api 
import pandas as pd
import numpy as np
from datetime import datetime

PARAMETERS = ['T2M', 'PRECTOT', 'WS10M', 'RH2M']  # temp, precip, wind, humidity

PARAMETERS = ['T2M', 'PRECTOT', 'WS10M', 'RH2M']  # temperature, precipitation, wind, humidity
HISTORICAL_YEARS = 20  # number of years to fetch


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

    # 1️⃣ Basic statistics for each climate variable
    for param in ['T2M', 'RH2M', 'WS10M', 'PRECTOTCORR']:
        stats[f"{param}_mean"] = dest_df[param].mean()
        stats[f"{param}_std"] = dest_df[param].std()
        stats[f"{param}_p90"] = np.percentile(dest_df[param], 90)
        stats[f"{param}_p10"] = np.percentile(dest_df[param], 10)

    # 2️⃣ Daily Discomfort Index (DI) using proven formula
    # DI = T - 0.55 * (1 - RH/100) * (T - 14.5)
    DI_daily = dest_df['T2M'] - 0.55 * (1 - dest_df['RH2M']/100) * (dest_df['T2M'] - 14.5)
    stats['DI_mean'] = DI_daily.mean()
    stats['DI_std'] = DI_daily.std()

    # 3️⃣ Precipitation probability
    stats['precip_prob'] = (dest_df['PRECTOTCORR'] > 0).mean()

    # 4️⃣ Wind mean
    stats['wind_mean'] = dest_df['WS10M'].mean()

    # 5️⃣ Optional: Relative to origin stats (if provided)
    if origin_stats:
        stats['T2M_relative'] = stats['T2M_mean'] - origin_stats.get('T2M_mean', stats['T2M_mean'])
        stats['RH2M_relative'] = stats['RH2M_mean'] - origin_stats.get('RH2M_mean', stats['RH2M_mean'])
        stats['DI_relative'] = stats['DI_mean'] - origin_stats.get('DI_mean', stats['DI_mean'])
    else:
        stats['T2M_relative'] = 0
        stats['RH2M_relative'] = 0
        stats['DI_relative'] = 0

    return stats



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
    # Define baseline preferences for each activity (T2M in °C, DI preferred range)
    activity_baselines = {
        'fishing': {'DI_mean': (18, 26)},       # mild and comfortable
        'hiking': {'DI_mean': (15, 28)},        # broader comfort
        'festival': {'DI_mean': (20, 30)},      # social outdoor events
        'generic': {'DI_mean': (16, 28)},       # generic default
    }

    baseline = activity_baselines.get(activity, activity_baselines['generic'])
    di_min, di_max = baseline['DI_mean']

    di = destination_stats.get('DI_mean', 25)

    # Weight based on DI within preferred range (linear scaling)
    if di < di_min:
        weight = max(0, 1 - (di_min - di) / 10)  # colder than optimal
    elif di > di_max:
        weight = max(0, 1 - (di - di_max) / 10)  # hotter than optimal
    else:
        weight = 1.0  # optimal range

    # Optionally adjust for extreme precipitation
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

    # 1️⃣ Normalize subscores (0 = worst, 1 = best)
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
def analysis(origin_lat, origin_lon, dest_lat, dest_lon, start_date, end_date, activities=None):
    """
    Run full climate + activity comfort analysis.
    
    Args:
        origin_lat, origin_lon: User's origin coordinates
        dest_lat, dest_lon: Destination coordinates
        start_date, end_date: YYYYMMDD
        activities (list): List of activity strings (optional)
    
    Returns:
        dict: {activity: comfort_index_dict, ...}
    """
    if activities is None:
        activities = ['fishing', 'hiking', 'festival', 'generic']

    # 1️⃣ Fetch climate data
    origin_df = fetch_origin_climate(origin_lat, origin_lon, start_date, end_date)
    dest_df = fetch_destination_historical(dest_lat, dest_lon, start_date, end_date)

    # 2️⃣ Compute statistics
    origin_stats = compute_statistics(origin_df) if not origin_df.empty else None
    dest_stats = compute_statistics(dest_df, origin_stats=origin_stats)

    results = {}

    # 3️⃣ Compute activity weights and comfort indices
    for activity in activities:
        weight = activity_weights(dest_stats, activity)
        adaptation_penalty = 0.2  # example, could be dynamic later
        comfort_index = compute_comfort_index(dest_stats, weight, adaptation_penalty)
        results[activity] = comfort_index

    return results


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
