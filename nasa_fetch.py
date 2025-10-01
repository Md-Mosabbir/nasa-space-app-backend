import requests
import pandas as pd
import json
from datetime import datetime

# ==== User Inputs ====
target_date = "20260210"  # YYYYMMDD format
years_back = 15           # Number of years to look back
latitude = 23.8041
longitude = 90.4152
parameter = "WS10M"         # Example: "T2M" = Temperature, "WS10M" = Windspeed

# ==== Extract MM-DD from target ====
target_mmdd = target_date[4:]  # "1220"
target_day = datetime.strptime(target_date, "%Y%m%d").strftime("%m-%d")

# ==== Current Year and Range ====
current_year = datetime.now().year
start_year = current_year - years_back
end_year = current_year

results = []

for year in range(start_year, end_year + 1):
    # Build API request for full year of daily data
    requests_url = (
        f"https://power.larc.nasa.gov/api/temporal/daily/point?"
        f"start={year}0101&end={year}1231&latitude={latitude}&longitude={longitude}"
        f"&community=ag&parameters={parameter}&format=json&units=metric"
        f"&user=TEMP&header=true&time-standard=utc"
    )

    response = requests.get(requests_url, timeout=40).json()

    try:
        data_dict = response["properties"]["parameter"][parameter]

        # Convert to DataFrame
        df = pd.DataFrame(list(data_dict.items()), columns=["date", parameter])
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

        # Find row with same MM-DD
        df["mmdd"] = df["date"].dt.strftime("%m-%d")
        match = df[df["mmdd"] == target_day]

        if not match.empty:
            results.append({
                "date": datetime(year, int(target_date[4:6]), int(target_date[6:])).strftime("%Y-%m-%d"),
                "value": float(match[parameter].values[0])
            })

    except KeyError:
        print(f"⚠️ No data for {year}")

# ==== Save results ====
output = {
    "target_date": target_date,
    "parameter": parameter,
    "data": results  # [{date: YYYY-MM-DD, value: ...}, ...]
}

with open("nasa_probability.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"✅ Data saved for {len(results)} years → nasa_probability.json")