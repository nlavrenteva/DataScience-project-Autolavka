import os
import sys
import numpy as np
import pandas as pd

BASE_LAT, BASE_LON = 51.73, 36.19

RAW_PATH = "osm_villages_raw.csv"
OUT_PATH = "villages.csv"


def load_data(filepath):
    data = pd.read_csv(filepath, sep=",")
    data = data.rename(columns={"@id": "osm_id", "@lat": "lat", "@lon": "lon"})
    needed_cols = [c for c in ["osm_id", "lat", "lon", "name", "place", "population"] if c in data.columns]
    return data[needed_cols].copy()


def clean_data(df):
    before_rows = len(df)
    df = df.dropna(subset=["lat", "lon", "name"])
    df = df.drop_duplicates(subset=["name", "lat", "lon"])
    df = df[df["name"].str.len() > 1]
    df["population"] = pd.to_numeric(df["population"], errors="coerce")
    return df.reset_index(drop=True)


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    lat1_rad, lat2_rad = np.radians(lat1), np.radians(lat2)
    dlat = lat2_rad - lat1_rad
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def fill_population(df, neighbors=5):
    known_pop = df[df["population"].notna()].copy()
    missing_idx = df[df["population"].isna()].index

    if known_pop.empty:
        df["population"] = df["population"].fillna(100)
        return df

    for idx in missing_idx:
        lat, lon = df.loc[idx, ["lat", "lon"]]
        dist = haversine_km(lat, lon, known_pop["lat"].values, known_pop["lon"].values)
        closest = np.argsort(dist)[:neighbors]
        df.at[idx, "population"] = known_pop.iloc[closest]["population"].mean()

    df["population"] = df["population"].clip(lower=10).round().astype(int)
    return df


def estimate_pct_pensioners(pop):
    val = 28 + 30 * np.exp(-pop / 500)
    return val.clip(20, 70).round(1)


def estimate_avg_income(distance_to_moscow, pension_share):
    base = 22000
    moscow_effect = 14000 * np.exp(-distance_to_moscow / 250)
    pension_effect = (pension_share - 30) * 100
    income = base + moscow_effect - pension_effect
    return income.clip(15000, 60000).round().astype(int)


def estimate_has_shop(df):
    rng = np.random.default_rng(42)
    pop = df["population"].clip(lower=10)
    logits = (np.log(pop) - 5.5) * 1.2
    prob = 1 / (1 + np.exp(-logits))
    return (rng.random(len(df)) < prob).astype(int)

df = load_data(RAW_PATH)
df = clean_data(df)
df = fill_population(df)

df["dist_to_city_km"] = np.round(haversine_km(df["lat"].values, df["lon"].values, BASE_LAT, BASE_LON), 1)

dist_moscow = haversine_km(df["lat"].values, df["lon"].values, 55.75, 37.62)

df["pct_pensioners"] = estimate_pct_pensioners(df["population"])
df["avg_income"] = estimate_avg_income(dist_moscow, df["pct_pensioners"])
df["has_shop"] = estimate_has_shop(df)

df = df.rename(columns={"name": "village"})

cols = ["village", "population", "lat", "lon", "pct_pensioners", "avg_income", "has_shop", "dist_to_city_km"]
df[cols].to_csv(OUT_PATH, index=False)def fill_population(df, neighbors=5):
    known_pop = df[df["population"].notna()].copy()
    missing_idx = df[df["population"].isna()].index

    if known_pop.empty:
        df["population"] = df["population"].fillna(100)
        return df

    for idx in missing_idx:
        lat, lon = df.loc[idx, ["lat", "lon"]]
        dist = haversine_km(lat, lon, known_pop["lat"].values, known_pop["lon"].values)
        closest = np.argsort(dist)[:neighbors]
        df.at[idx, "population"] = known_pop.iloc[closest]["population"].mean()

    df["population"] = df["population"].clip(lower=10).round().astype(int)
    return df


def estimate_pct_pensioners(pop):
    val = 28 + 30 * np.exp(-pop / 500)
    return val.clip(20, 70).round(1)


def estimate_avg_income(distance_to_moscow, pension_share):
    base = 22000
    moscow_effect = 14000 * np.exp(-distance_to_moscow / 250)
    pension_effect = (pension_share - 30) * 100
    income = base + moscow_effect - pension_effect
    return income.clip(15000, 60000).round().astype(int)


def estimate_has_shop(df):
    rng = np.random.default_rng(42)
    pop = df["population"].clip(lower=10)
    logits = (np.log(pop) - 5.5) * 1.2
    prob = 1 / (1 + np.exp(-logits))
    return (rng.random(len(df)) < prob).astype(int)

df = load_data(RAW_PATH)
df = clean_data(df)
df = fill_population(df)

df["dist_to_city_km"] = np.round(haversine_km(df["lat"].values, df["lon"].values, BASE_LAT, BASE_LON), 1)

dist_moscow = haversine_km(df["lat"].values, df["lon"].values, 55.75, 37.62)

df["pct_pensioners"] = estimate_pct_pensioners(df["population"])
df["avg_income"] = estimate_avg_income(dist_moscow, df["pct_pensioners"])
df["has_shop"] = estimate_has_shop(df)

df = df.rename(columns={"name": "village"})

cols = ["village", "population", "lat", "lon", "pct_pensioners", "avg_income", "has_shop", "dist_to_city_km"]
df[cols].to_csv(OUT_PATH, index=False)
