import numpy as np
import pandas as pd
from datetime import datetime
from ..controllers.power_prediction import compute_statistics

# -------------------- Helper: Generate synthetic 10 years of weather data --------------------
def generate_mock_weather(start_year, end_year, T_mean, RH_mean, WS_mean, PRE_mean, noise=1.0):
    """
    Generate synthetic daily weather data for multiple years.
    """
    frames = []
    for year in range(start_year, end_year + 1):
        dates = pd.date_range(f"{year}-01-01", f"{year}-12-31")
        df = pd.DataFrame({
            "T2M": np.random.normal(T_mean, noise, len(dates)),
            "RH2M": np.random.normal(RH_mean, noise, len(dates)),
            "WS10M": np.random.normal(WS_mean, noise, len(dates)),
            "PRECTOTCORR": np.random.normal(PRE_mean, noise, len(dates))
        }, index=dates)
        frames.append(df)
    return pd.concat(frames)

# -------------------- Setup --------------------
current_year = datetime.now().year
start_year = current_year - 10  # past 10 years

# -------------------- Test 1: Hot climate --------------------
hot_df = generate_mock_weather(start_year, current_year, T_mean=35, RH_mean=70, WS_mean=1, PRE_mean=0.5)
hot_stats = compute_statistics(hot_df)
print("=== Heat Test (10 years) ===")
print(f"Median DI: {hot_stats['DI_median']:.2f}, Median Precipitation Probability: {hot_stats['precip_prob']:.2f}")

# -------------------- Test 2: Cold climate --------------------
cold_df = generate_mock_weather(start_year, current_year, T_mean=5, RH_mean=80, WS_mean=2, PRE_mean=0)
cold_stats = compute_statistics(cold_df)
print("=== Cold Test (10 years) ===")
print(f"Median DI: {cold_stats['DI_median']:.2f}, Median Precipitation Probability: {cold_stats['precip_prob']:.2f}")

# -------------------- Test 3: Windy climate --------------------
windy_df = generate_mock_weather(start_year, current_year, T_mean=20, RH_mean=50, WS_mean=10, PRE_mean=0)
windy_stats = compute_statistics(windy_df)
print("=== Wind Test (10 years) ===")
print(f"Median Wind: {windy_stats['wind_median']:.2f}, Median DI: {windy_stats['DI_median']:.2f}")

# -------------------- Test 4: Rainy climate --------------------
rainy_df = generate_mock_weather(start_year, current_year, T_mean=22, RH_mean=90, WS_mean=2, PRE_mean=15)
rainy_stats = compute_statistics(rainy_df)
print("=== Rain Test (10 years) ===")
print(f"Median Precip Prob: {rainy_stats['precip_prob']:.2f}, Median DI: {rainy_stats['DI_median']:.2f}")

# -------------------- Test 5: Extreme climate --------------------
risky_df = generate_mock_weather(start_year, current_year, T_mean=38, RH_mean=95, WS_mean=8, PRE_mean=20)
risky_stats = compute_statistics(risky_df)
print("=== Risky Test (10 years) ===")
print(f"Median DI: {risky_stats['DI_median']:.2f}, Median Wind: {risky_stats['wind_median']:.2f}, Median Precip Prob: {risky_stats['precip_prob']:.2f}")
