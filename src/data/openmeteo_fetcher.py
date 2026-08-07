import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class OpenMeteoAgriFetcher:
    """
    Fetches live real-world high-resolution agricultural climate and soil moisture data
    from the Open-Meteo Historical & Forecast Weather API.
    """
    def __init__(self):
        self.base_url = "https://archive-api.open-meteo.com/v1/archive"
        self.forecast_url = "https://api.open-meteo.com/v1/forecast"

    def fetch_field_climate_history(self, lat, lon, start_date, end_date):
        """
        Fetches daily temperature, precipitation, and multi-depth soil moisture
        for specific field coordinates (latitude, longitude).
        """
        print(f"Fetching live weather & soil moisture data for coordinates ({lat}, {lon}) from {start_date} to {end_date}...")
        
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": [
                "temperature_2m_max",
                "temperature_2m_mean",
                "precipitation_sum",
                "soil_moisture_0_to_7cm_mean",
                "soil_moisture_7_to_28cm_mean"
            ],
            "timezone": "auto"
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            daily_data = data.get("daily", {})
            df = pd.DataFrame({
                "date": daily_data.get("time", []),
                "temp_max_c": daily_data.get("temperature_2m_max", []),
                "temp_mean_c": daily_data.get("temperature_2m_mean", []),
                "precip_mm": daily_data.get("precipitation_sum", []),
                "soil_m_shallow": daily_data.get("soil_moisture_0_to_7cm_mean", []),
                "soil_m_deep": daily_data.get("soil_moisture_7_to_28cm_mean", [])
            })
            
            # Convert volumetric soil moisture (m³/m³) to percentage (%)
            df["soil_moisture_%"] = df["soil_m_deep"].apply(lambda x: round(x * 100, 2) if pd.notnull(x) else 20.0)
            df["date"] = pd.to_datetime(df["date"])
            
            print(f"Successfully fetched {len(df)} days of live weather & soil data.")
            return df
            
        except Exception as e:
            print(f"Error fetching Open-Meteo data: {e}")
            return None

    def aggregate_to_sentinel_cadence(self, df_daily, interval_days=5):
        """
        Aggregates daily weather data into 5-day intervals matching Sentinel-2 satellite pass cadence.
        """
        if df_daily is None or df_daily.empty:
            return None
            
        df_daily = df_daily.sort_values("date").reset_index(drop=True)
        df_daily["step"] = df_daily.index // interval_days
        
        agg_df = df_daily.groupby("step").agg({
            "date": "first",
            "temp_mean_c": "mean",
            "temp_max_c": "max",
            "precip_mm": "sum",
            "soil_moisture_%": "mean",
            "soil_m_shallow": "mean",
            "soil_m_deep": "mean"
        }).reset_index()
        
        agg_df.rename(columns={"date": "step_start_date"}, inplace=True)
        return agg_df

if __name__ == "__main__":
    # Test with coordinates for a wheat field in Konya, Turkey (Lat: 37.87, Lon: 32.49)
    fetcher = OpenMeteoAgriFetcher()
    end_dt = datetime.now().strftime("%Y-%m-%d")
    start_dt = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    
    df_live = fetcher.fetch_field_climate_history(lat=37.87, lon=32.49, start_date=start_dt, end_date=end_dt)
    if df_live is not None:
        df_sentinel_weather = fetcher.aggregate_to_sentinel_cadence(df_live, interval_days=5)
        os.makedirs("data", exist_ok=True)
        df_sentinel_weather.to_csv("data/live_konya_climate.csv", index=False)
        print("\nLive Konya Climate Sample (5-Day Aggregated for Sentinel-2):")
        print(df_sentinel_weather.head().to_string(index=False))
