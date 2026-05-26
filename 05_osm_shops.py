class OSMShopsParser:
    ENDPOINTS = [
        'https://overpass-api.de/api/interpreter',
        'https://overpass.kumi.systems/api/interpreter',
        'https://overpass.openstreetmap.ru/api/interpreter',
    ]
    SHOP_TYPES = 'supermarket|convenience|greengrocer|bakery|butcher|general|kiosk'

    def __init__(self, timeout=180):
        self.timeout = timeout
        self.headers = {'User-Agent': 'AutolavkaProject/1.0'}

    def build_query(self, bbox):
        south, west, north, east = bbox
        return f'''
[out:csv(::id,::lat,::lon,name,shop;true;",")][timeout:120];
(
  node["shop"~"{self.SHOP_TYPES}"]({south},{west},{north},{east});
);
out;
'''

    def fetch_region(self, region, bbox):
        query = self.build_query(bbox)
        for endpoint in self.ENDPOINTS:
            try:
                r = requests.post(endpoint, data={'data': query},
                                  headers=self.headers, timeout=self.timeout)
                if r.status_code == 200:
                    df = pd.read_csv(BytesIO(r.content))
                    df = df.rename(columns={'@lat': 'lat', '@lon': 'lon', 'shop': 'shop_type'})
                    df = df.dropna(subset=['lat', 'lon'])
                    print(f'{region}: {len(df)} магазинов')
                    return df
            except Exception:
                continue
        return pd.DataFrame()

    def fetch_many(self, bboxes, max_workers=2):
        tables = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(self.fetch_region, name, bbox): name
                       for name, bbox in bboxes.items()}
            for f in as_completed(futures):
                df = f.result()
                if not df.empty:
                    tables.append(df)
        if not tables:
            return pd.DataFrame()
        out = pd.concat(tables, ignore_index=True)
        if '@id' in out.columns:
            out = out.drop_duplicates(subset=['@id'])
        return out.reset_index(drop=True)


shops_df = OSMShopsParser().fetch_many(OSMVillagesParser.BBOXES)
shops_df.to_csv('osm_shops_raw.csv', index=False)
print(f'\nВсего магазинов: {len(shops_df)}')
print(shops_df['shop_type'].value_counts())
shops_df.head()

class ShopsFeatureBuilder:
    def __init__(self, radius_km=3.0):
        self.radius_km = radius_km

    @staticmethod
    def haversine_vec(lat1, lon1, lat2_vec, lon2_vec):
        R = 6371
        lat1_r, lat2_r = np.radians(lat1), np.radians(lat2_vec)
        dlat = lat2_r - lat1_r
        dlon = np.radians(lon2_vec - lon1)
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
        return 2 * R * np.arcsin(np.sqrt(a))

    def build(self, villages_df, shops_df):
        n = len(villages_df)
        n_shops = np.zeros(n, dtype=int)
        nearest = np.full(n, 15.0)
        has_super = np.zeros(n, dtype=int)
        s_lat, s_lon = shops_df['lat'].values, shops_df['lon'].values
        s_type = shops_df['shop_type'].values
        for i, (_, v) in enumerate(villages_df.iterrows()):
            d = self.haversine_vec(v['lat'], v['lon'], s_lat, s_lon)
            in_r = d <= self.radius_km
            n_shops[i] = in_r.sum()
            if d.size:
                nearest[i] = d.min()
            if in_r.any():
                has_super[i] = int((s_type[in_r] == 'supermarket').any())
        res = villages_df.copy()
        res['n_shops_nearby'] = n_shops
        res['nearest_shop_km'] = np.round(nearest, 2)
        res['has_supermarket_nearby'] = has_super
        res['has_any_shop_nearby'] = (res['n_shops_nearby'] > 0).astype(int)
        return res


villages = ShopsFeatureBuilder(radius_km=3.0).build(villages, shops_df)
villages.to_csv('villages.csv', index=False)

zero = villages['n_shops_nearby'] == 0
print(f'Деревень без магазинов в радиусе 3 км: {zero.sum()} ({zero.mean()*100:.1f}%)')
print(f'Медиана расстояния до ближайшего магазина: {villages["nearest_shop_km"].median():.2f} км')
print(villages['n_shops_nearby'].describe().round(2))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].hist(villages['n_shops_nearby'], bins=range(int(villages['n_shops_nearby'].max()) + 2),
             color='#4C9AFF', edgecolor='white')
axes[0].axvline(villages['n_shops_nearby'].median(), color='red', linestyle='--',
                label=f'Медиана: {villages["n_shops_nearby"].median():.0f}')
axes[0].set_xlabel('Магазинов в радиусе 3 км'); axes[0].set_ylabel('Деревень')
axes[0].set_title('Магазинов рядом с деревней'); axes[0].legend()

axes[1].hist(villages['nearest_shop_km'].clip(upper=15), bins=30, color='#FF8B94', edgecolor='white')
axes[1].axvline(3, color='green', linestyle='--', label='Пешая доступность (3 км)')
axes[1].set_xlabel('Расстояние до магазина, км'); axes[1].set_ylabel('Деревень')
axes[1].set_title('Удалённость до ближайшего магазина'); axes[1].legend()

bins = pd.cut(villages['population'], bins=[0, 50, 100, 300, 1000, 100000],
              labels=['<50', '50-100', '100-300', '300-1000', '1000+'])
agg = villages.groupby(bins, observed=True)['n_shops_nearby'].mean()
axes[2].bar(agg.index.astype(str), agg.values, color='#88D498')
axes[2].set_xlabel('Население деревни'); axes[2].set_ylabel('Среднее число магазинов рядом')
axes[2].set_title('Больше деревня — больше магазинов рядом')

plt.tight_layout()
plt.savefig('shops_eda.png', dpi=110, bbox_inches='tight')
plt.show()
