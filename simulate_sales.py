import numpy as np
import pandas as pd


def simulate_sales(dataframe, trips_count=8, seed=42):
    rng = np.random.default_rng(seed)
    num_villages = len(dataframe)

    loyalty_factors = rng.lognormal(mean=0, sigma=0.4, size=num_villages)

    road_quality_raw = 1 - dataframe["dist_to_city_km"].values / 400
    road_quality_noise = rng.normal(0, 0.15, size=num_villages)
    road_quality = road_quality_raw + road_quality_noise
    road_quality = np.clip(road_quality, 0.3, 1.0)

    trips_data = []

    for trip_num in range(trips_count):
        seasonal_multiplier = rng.uniform(0.7, 1.3)
        bad_weather = rng.random() < 0.15

        active_rates = estimate_active_rate(dataframe)
        avg_checks = estimate_avg_check(dataframe)

        revenues = (
            dataframe["population"].values
            * active_rates
            * avg_checks
            * loyalty_factors
            * road_quality
            * seasonal_multiplier
        )

        if bad_weather:
            revenues = revenues * 0.5

        sales_noise = rng.lognormal(mean=0, sigma=0.25, size=num_villages)
        revenues = revenues * sales_noise
        revenues = np.clip(revenues, 0, None).round(0)

        trip_frame = pd.DataFrame({
            "village": dataframe["village"].values,
            "trip_id": trip_num,
            "actual_revenue": revenues
        })

        trips_data.append(trip_frame)

    return pd.concat(trips_data, ignore_index=True)


def estimate_active_rate(dataframe):
    pensioners_pct = dataframe["pct_pensioners"].values
    shops_present = dataframe["has_shop"].values

    base_rate = 0.20 + 0.0035 * pensioners_pct
    penalty_for_shop = shops_present * (0.5 - 0.003 * pensioners_pct)

    activity_rate = base_rate - penalty_for_shop
    return np.clip(activity_rate, 0.02, 0.8)


def estimate_avg_check(dataframe):
    avg_income_vals = dataframe["avg_income"].values
    return 150 + 200 * np.log1p(avg_income_vals / 5000)


def aggregate_by_village(sales_df):
    grouped = sales_df.groupby("village").agg(
        avg_revenue=("actual_revenue", "mean"),
        median_revenue=("actual_revenue", "median"),
        std_revenue=("actual_revenue", "std"),
        n_trips=("actual_revenue", "count")
    ).reset_index()

    for col in ["avg_revenue", "median_revenue", "std_revenue"]:
        grouped[col] = grouped[col].round(0)

    return grouped

villages_df = pd.read_csv("villages.csv")
sales_history = simulate_sales(villages_df, trips_count=8)
village_summary = aggregate_by_village(sales_history)
village_summary.to_csv("village_revenue.csv", index=False)
print("\nСтатистика выручки:")
print(village_summary["avg_revenue"].describe().round(0))
print("\nТоп-5 деревень по средней выручке:")
print(village_summary.nlargest(5, "avg_revenue").to_string(index=False))
