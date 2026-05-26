class OSMVillagesParser:
    ENDPOINTS = [
        'https://overpass-api.de/api/interpreter',
        'https://overpass.kumi.systems/api/interpreter',
        'https://overpass.openstreetmap.ru/api/interpreter',
    ]
    BBOXES = {
        'Курская область':   (51.2, 34.0, 52.5, 38.5),
        'Орловская область': (51.8, 34.7, 53.7, 38.1),
        'Брянская область':  (51.8, 31.0, 54.0, 35.5),
        'Тульская область':  (53.0, 35.8, 54.8, 39.0),
    }

    def __init__(self, timeout=180):
        self.timeout = timeout
        self.headers = {'User-Agent': 'AutolavkaProject/1.0'}

    def build_query(self, bbox):
        south, west, north, east = bbox
        return f'''
[out:csv(::id,::lat,::lon,name,place,population,"addr:region";true;",")][timeout:120];
(
  node["place"~"village|hamlet|isolated_dwelling"]({south},{west},{north},{east});
);
out;
'''

    def fetch_region(self, region):
        query = self.build_query(self.BBOXES[region])
        for endpoint in self.ENDPOINTS:
            try:
                r = requests.post(endpoint, data={'data': query},
                                  headers=self.headers, timeout=self.timeout)
                if r.status_code == 200:
                    df = pd.read_csv(BytesIO(r.content))
                    df['source_region'] = region
                    print(f'{region}: {len(df)} деревень')
                    return df
            except Exception:
                continue
        print(f'{region}: не удалось загрузить')
        return pd.DataFrame()

    def fetch_many(self, regions, max_workers=2):
        tables = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(self.fetch_region, r) for r in regions]
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

parser = OSMVillagesParser()
raw = parser.fetch_many(list(OSMVillagesParser.BBOXES))

# bbox захватывает соседние территории — оставляем только целевые регионы
target_regions = set(OSMVillagesParser.BBOXES)
if 'addr:region' in raw.columns:
    wrong = raw['addr:region'].notna() & ~raw['addr:region'].isin(target_regions)
    raw = raw[~wrong].reset_index(drop=True)

print(f'Всего сырых деревень: {len(raw)}')
raw.head()
