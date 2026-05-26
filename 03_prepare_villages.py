class VillagesPreparer:
    EARTH_RADIUS_KM = 6371

    def __init__(self, base_lat=51.73, base_lon=36.19, moscow_lat=55.75, moscow_lon=37.62):
        self.base_lat, self.base_lon = base_lat, base_lon
        self.moscow_lat, self.moscow_lon = moscow_lat, moscow_lon
        self.df = None

    @staticmethod
    def haversine_km(lat1, lon1, lat2, lon2):
        R = VillagesPreparer.EARTH_RADIUS_KM
        lat1_r, lat2_r = np.radians(lat1), np.radians(lat2)
        dlat = lat2_r - lat1_r
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
        return 2 * R * np.arcsin(np.sqrt(a))

    def load(self, df):
        df = df.rename(columns={'@id': 'osm_id', '@lat': 'lat', '@lon': 'lon'})
        cols = [c for c in ['osm_id', 'lat', 'lon', 'name', 'place', 'population'] if c in df.columns]
        self.df = df[cols].copy()
        return self

    def clean(self):
        df = self.df.dropna(subset=['lat', 'lon', 'name'])
        df = df.drop_duplicates(subset=['name', 'lat', 'lon'])
        df = df[df['name'].str.len() > 1]
        df['population'] = pd.to_numeric(df['population'], errors='coerce')
        self.df = df.reset_index(drop=True)
        return self

    def fill_population(self, neighbors=5):
        df = self.df
        known = df[df['population'].notna()]
        if known.empty:
            df['population'] = df['population'].fillna(100)
        else:
            for idx in df[df['population'].isna()].index:
                dist = self.haversine_km(df.loc[idx, 'lat'], df.loc[idx, 'lon'],
                                         known['lat'].values, known['lon'].values)
                nearest = known.iloc[np.argsort(dist)[:neighbors]]
                df.at[idx, 'population'] = nearest['population'].mean()
        df['population'] = df['population'].clip(lower=10).round().astype(int)
        self.df = df
        return self

    @staticmethod
    def estimate_pct_pensioners(pop):
        return (28 + 30 * np.exp(-pop / 500)).clip(20, 70).round(1)

    @staticmethod
    def estimate_avg_income(dist_to_moscow, pension_share):
        income = 22000 + 14000 * np.exp(-dist_to_moscow / 250) - (pension_share - 30) * 100
        return income.clip(15000, 60000).round().astype(int)

    def engineer_features(self):
        df = self.df
        df['dist_to_city_km'] = np.round(
            self.haversine_km(df['lat'].values, df['lon'].values, self.base_lat, self.base_lon), 1)
        dist_moscow = self.haversine_km(df['lat'].values, df['lon'].values,
                                        self.moscow_lat, self.moscow_lon)
        df['pct_pensioners'] = self.estimate_pct_pensioners(df['population'])
        df['avg_income'] = self.estimate_avg_income(dist_moscow, df['pct_pensioners'])
        df = df.rename(columns={'name': 'village'})
        self.df = df[['village', 'population', 'lat', 'lon',
                      'pct_pensioners', 'avg_income', 'dist_to_city_km']]
        return self

    def save(self, path='villages.csv'):
        self.df.to_csv(path, index=False)
        return self

    def get(self):
        return self.df


villages = (VillagesPreparer()
            .load(raw)
            .clean()
            .fill_population()
            .engineer_features()
            .save()
            .get())
print(f'Подготовлено {len(villages)} деревень')
villages.head()
