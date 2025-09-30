from groq import Groq


data = {
  "origin": {
    "latitude": 23.8103,
    "longitude": 90.4125,
    "statistics": {
      "T2M_mean": 27.4465,
      "T2M_std": 0.8137209470457406,
      "T2M_p90": 28.336,
      "T2M_p10": 26.308,
      "RH2M_mean": 86.66349999999998,
      "RH2M_std": 5.349050232502076,
      "RH2M_p90": 93.07799999999999,
      "RH2M_p10": 78.81299999999999,
      "WS10M_mean": 2.0725,
      "WS10M_std": 1.1184669062332524,
      "WS10M_p90": 3.303,
      "WS10M_p10": 1.208,
      "PRECTOTCORR_mean": 12.505833333333333,
      "PRECTOTCORR_std": 17.24358935575933,
      "PRECTOTCORR_p90": 24.549000000000007,
      "PRECTOTCORR_p10": 0.4480000000000001,
      "DI_mean": 26.482762014166664,
      "DI_std": 0.617562820732783,
      "precip_prob": 0.95,
      "wind_mean": 2.0725,
      "T2M_relative": 0,
      "RH2M_relative": 0,
      "DI_relative": 0
    }
  },
  "destination": {
    "latitude": 29.7604,
    "longitude": -95.3698,
    "statistics": {
      "T2M_mean": 24.258333333333333,
      "T2M_std": 2.845509794167042,
      "T2M_p90": 27.439,
      "T2M_p10": 19.941,
      "RH2M_mean": 70.438,
      "RH2M_std": 11.824583527263036,
      "RH2M_p90": 85.24799999999999,
      "RH2M_p10": 57.706,
      "WS10M_mean": 2.783,
      "WS10M_std": 0.9397913580205818,
      "WS10M_p90": 4.023,
      "WS10M_p10": 1.709,
      "PRECTOTCORR_mean": 1.6054999999999997,
      "PRECTOTCORR_std": 4.3382947618480205,
      "PRECTOTCORR_p90": 3.558000000000004,
      "PRECTOTCORR_p10": 0,
      "DI_mean": 22.764769814999998,
      "DI_std": 2.6682307484023093,
      "precip_prob": 0.5666666666666667,
      "wind_mean": 2.783,
      "T2M_relative": -3.1881666666666675,
      "RH2M_relative": -16.225499999999982,
      "DI_relative": -3.717992199166666
    }
  },
  "activities": {
    "hiking": {
      "weight": 0.7166666666666667,
      "adaptation_penalty": 0.2,
      "comfort_index": {
        "score": 30,
        "category": "Poor",
        "color": "orange",
        "main_factors": [
          "Precipitation"
        ]
      },
      "risks": {
        "hot_risk": 0,
        "cold_risk": 2,
        "rain_risk": 15,
        "wind_risk": 5
      }
    },
    "fishing": {
      "weight": 0.7166666666666667,
      "adaptation_penalty": 0.2,
      "comfort_index": {
        "score": 30,
        "category": "Poor",
        "color": "orange",
        "main_factors": [
          "Precipitation"
        ]
      },
      "risks": {
        "hot_risk": 0,
        "cold_risk": 2,
        "rain_risk": 15,
        "wind_risk": 5
      }
    },
    "festival": {
      "weight": 0.7166666666666667,
      "adaptation_penalty": 0.2,
      "comfort_index": {
        "score": 30,
        "category": "Poor",
        "color": "orange",
        "main_factors": [
          "Precipitation"
        ]
      },
      "risks": {
        "hot_risk": 0,
        "cold_risk": 2,
        "rain_risk": 15,
        "wind_risk": 5
      }
    }
  },
  "meta": {
    "start_date": "20251005",
    "end_date": "20251007",
    "historical_years": 20,
    "notes": "DI calculated using NASA POWER data; missing precipitation treated as 0"
  }
}

client = Groq()

prompt = f"""
You are a friendly weather assistant.
Here is the destination weather for {data['meta']['start_date']} to {data['meta']['end_date']}:
Temperature mean: {data['destination']['statistics']['T2M_mean']} °C
Discomfort Index: {data['destination']['statistics']['DI_mean']}
Rain probability: {data['destination']['statistics']['precip_prob'] * 100}%
Wind speed: {data['destination']['statistics']['wind_mean']} m/s

Activities: {', '.join(data['activities'].keys())}

Provide concise, actionable advice for each activity. Mention main risks and tips for comfort.
"""

completion = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.7,
    max_completion_tokens=400
)

print(completion.choices[0].message.content)