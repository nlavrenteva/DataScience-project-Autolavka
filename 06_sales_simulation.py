class SalesSimulator:
    def __init__(self, trips_count=8, seed=42):
        self.trips_count = trips_count
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def estimate_active_rate(df):
        pens = df['pct_pensioners'].values
        n_shops = df['n_shops_nearby'].values
        nearest = df['nearest_shop_km'].values
        base = 0.20 + 0.0035 * pens
        competition = 0.03 * n_shops * (1 - 0.003 * pens)
        distance_bonus = 0.015 * np.clip(nearest - 1, 0, 10)
        return np.clip(base - competition + distance_bonus, 0.02, 0.8)

    @staticmethod
    def estimate_avg_check(df):
        return 150 + 200 * np.log1p(df['avg_income'].values / 5000)

    def simulate(self, df):
        n = len(df)
        loyalty = self.rng.lognormal(0, 0.4, n)
        road_q = np.clip(1 - df['dist_to_city_km'].values / 400 + self.rng.normal(0, 0.15, n), 0.3, 1.0)
        active = self.estimate_active_rate(df)
        check = self.estimate_avg_check(df)
        trips = []
        for t in range(self.trips_count):
            seasonal = self.rng.uniform(0.7, 1.3)
            rev = df['population'].values * active * check * loyalty * road_q * seasonal
            if self.rng.random() < 0.15:
                rev *= 0.5
            rev *= self.rng.lognormal(0, 0.25, n)
            trips.append(pd.DataFrame({'village': df['village'].values, 'trip_id': t,
                                       'actual_revenue': np.clip(rev, 0, None).round(0)}))
        return pd.concat(trips, ignore_index=True)

    @staticmethod
    def aggregate(sales):
        agg = sales.groupby('village').agg(
            avg_revenue=('actual_revenue', 'mean'),
            median_revenue=('actual_revenue', 'median'),
        ).reset_index()
        agg[['avg_revenue', 'median_revenue']] = agg[['avg_revenue', 'median_revenue']].round(0)
        return agg


sim = SalesSimulator(trips_count=8)
village_revenue = sim.aggregate(sim.simulate(villages))
village_revenue.to_csv('village_revenue.csv', index=False)

data = villages.merge(village_revenue, on='village')
print(data['avg_revenue'].describe().round(0))
data.head()

data['cost_per_visit'] = 200 * data['dist_to_city_km'] + 3000
data['is_profitable'] = (data['avg_revenue'] >= data['cost_per_visit']).astype(int)
print(f'Доля прибыльных деревень: {data["is_profitable"].mean()*100:.1f}% '
      f'({data["is_profitable"].sum()} из {len(data)})')

fig, ax = plt.subplots(figsize=(11, 6))
for flag, color, label in [(0, '#FF6B6B', 'Неприбыльные'), (1, '#51CF66', 'Прибыльные')]:
    sub = data[data['is_profitable'] == flag]
    ax.scatter(sub['dist_to_city_km'], sub['avg_revenue'], alpha=0.4, s=12,
               c=color, label=f'{label} ({len(sub)})')
xs = np.linspace(0, data['dist_to_city_km'].max(), 100)
ax.plot(xs, 200 * xs + 3000, 'k--', lw=2, label='Точка безубыточности')
ax.set_yscale('log')
ax.set_xlabel('Расстояние от базы')
ax.set_ylabel('Средняя выручка за поездку логарифмированная')
ax.set_title('Прибыльные и неприбыльные деревни')
ax.legend()
plt.savefig('cls_target.png', dpi=110, bbox_inches='tight')
plt.show()
