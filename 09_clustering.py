to_route = data[data['ml_include_pred'] == 1].copy().reset_index(drop=True)
print(f'Деревень для кластеризации: {len(to_route)}')

coords_sc = StandardScaler().fit_transform(to_route[['lat', 'lon']].values)

ks = range(2, 21)
inertias = [KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(coords_sc).inertia_
            for k in ks]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(ks, inertias, marker='o', color='#FF8B94', lw=2)
ax.axvline(8, color='green', linestyle='--', alpha=0.6, label='Выбрано: k=8')
ax.set_xlabel('Число кластеров k'); ax.set_ylabel('Inertia')
ax.set_title('Метод локтя для выбора числа зон'); ax.legend()
plt.savefig('clust_elbow.png', dpi=110, bbox_inches='tight')
plt.show()

N_CLUSTERS = 8

kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=20)
to_route['route_id'] = kmeans.fit_predict(coords_sc)

route_stats = (to_route.groupby('route_id').agg(
    n_villages=('village', 'count'),
    total_population=('population', 'sum'),
    total_predicted_revenue=('predicted_revenue', 'sum'),
    avg_dist_from_base=('dist_to_city_km', 'mean'),
).reset_index().sort_values('total_predicted_revenue', ascending=False))
route_stats['total_predicted_revenue'] = route_stats['total_predicted_revenue'].round(0)
route_stats['avg_dist_from_base'] = route_stats['avg_dist_from_base'].round(1)
print('Статистика по зонам обслуживания:')
print(route_stats.to_string(index=False))

fig, ax = plt.subplots(figsize=(13, 9))
palette = sns.color_palette('tab10', N_CLUSTERS)
for cid in range(N_CLUSTERS):
    sub = to_route[to_route['route_id'] == cid]
    ax.scatter(sub['lon'], sub['lat'], s=20, alpha=0.7, color=palette[cid],
               label=f'Зона {cid} ({len(sub)} дер.)')
ax.scatter([36.19], [51.73], s=400, c='black', marker='*', edgecolors='red',
           linewidths=2, label='База (Курск)', zorder=5)
ax.set_xlabel('Долгота'); ax.set_ylabel('Широта')
ax.set_title(f'Кластеризация {len(to_route)} деревень на {N_CLUSTERS} зон обслуживания')
ax.legend(loc='upper left', fontsize=9, ncol=2)
plt.savefig('clust_map.png', dpi=110, bbox_inches='tight')
plt.show()

(to_route[['village', 'lat', 'lon', 'population', 'avg_income', 'predicted_revenue', 'route_id']]
 .sort_values(['route_id', 'predicted_revenue'], ascending=[True, False])
 .to_csv('village_clusters.csv', index=False))
