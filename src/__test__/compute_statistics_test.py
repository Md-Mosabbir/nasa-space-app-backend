import numpy as np
import pandas as pd
from ..controllers.power_prediction import compute_statistics 

# -------------------- Helper to generate mock data --------------------
def generate_mock_weather(dates, T_mean, RH_mean, WS_mean, PRE_mean, noise=1.0):
    return pd.DataFrame({
        'T2M': np.random.normal(T_mean, noise, len(dates)),
        'RH2M': np.random.normal(RH_mean, noise, len(dates)),
        'WS10M': np.random.normal(WS_mean, noise, len(dates)),
        'PRECTOTCORR': np.random.normal(PRE_mean, noise, len(dates))
    }, index=dates)

# -------------------- Dates --------------------
dates = pd.date_range("2024-10-10", "2024-10-20")  # 11 days

# -------------------- Test 1: Hot weather --------------------
hot_df = generate_mock_weather(dates, T_mean=35, RH_mean=70, WS_mean=1, PRE_mean=0.5)
hot_stats = compute_statistics(hot_df)
print("=== Heat Test ===")
print(f"DI_mean: {hot_stats['DI_mean']:.2f}, precip_prob: {hot_stats['precip_prob']:.2f}")

# -------------------- Test 2: Cold weather --------------------
cold_df = generate_mock_weather(dates, T_mean=5, RH_mean=80, WS_mean=2, PRE_mean=0)
cold_stats = compute_statistics(cold_df)
print("=== Cold Test ===")
print(f"DI_mean: {cold_stats['DI_mean']:.2f}, precip_prob: {cold_stats['precip_prob']:.2f}")

# -------------------- Test 3: Windy weather --------------------
windy_df = generate_mock_weather(dates, T_mean=20, RH_mean=50, WS_mean=10, PRE_mean=0)
windy_stats = compute_statistics(windy_df)
print("=== Wind Test ===")
print(f"wind_mean: {windy_stats['wind_mean']:.2f}, DI_mean: {windy_stats['DI_mean']:.2f}")

# -------------------- Test 4: Rainy weather --------------------
rainy_df = generate_mock_weather(dates, T_mean=22, RH_mean=90, WS_mean=2, PRE_mean=15)
rainy_stats = compute_statistics(rainy_df)
print("=== Rain Test ===")
print(f"precip_prob: {rainy_stats['precip_prob']:.2f}, DI_mean: {rainy_stats['DI_mean']:.2f}")

# -------------------- Test 5: Risky/Extreme combination --------------------
risky_df = generate_mock_weather(dates, T_mean=38, RH_mean=95, WS_mean=8, PRE_mean=20)
risky_stats = compute_statistics(risky_df)
print("=== Risky Test ===")
print(f"DI_mean: {risky_stats['DI_mean']:.2f}, wind_mean: {risky_stats['wind_mean']:.2f}, precip_prob: {risky_stats['precip_prob']:.2f}")
