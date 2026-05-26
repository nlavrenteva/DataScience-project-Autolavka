rev_no = data[data['has_any_shop_nearby'] == 0]['avg_revenue']
rev_yes = data[data['has_any_shop_nearby'] == 1]['avg_revenue']
t_stat, p_value = stats.ttest_ind(rev_no, rev_yes, equal_var=False)

#Гипотеза 1: магазин в радиусе 3 км vs нет магазина c помощью t-теста
print(f'Выручка без магазина: {rev_no.mean():.0f} ₽ (n={len(rev_no)})')
print(f'Выручка с магазином: {rev_yes.mean():.0f} ₽ (n={len(rev_yes)})')
print(f' t = {t_stat:.2f}, p = {p_value:.6f}')
print('Вывод: ' + ('различие значимо (p < 0.05)' if p_value < 0.05 else 'различие не значимо (p >= 0.05)'))

fig, ax = plt.subplots(figsize=(8, 5))
bp = ax.boxplot([rev_no.clip(upper=rev_no.quantile(0.95)),
                 rev_yes.clip(upper=rev_yes.quantile(0.95))],
                labels=['Нет магазина', 'Есть магазин'], patch_artist=True, widths=0.5)
for patch, color in zip(bp['boxes'], ['#88D498', '#FF8B94']):
    patch.set_facecolor(color)
ax.set_ylabel('Средняя выручка за поездку')
ax.set_title(f'Выручка и наличие магазина рядом')
plt.savefig('hyp_shop.png', dpi=110, bbox_inches='tight')
plt.show()

data['revenue_per_capita'] = data['avg_revenue'] / data['population']
corr, p_corr = stats.spearmanr(data['pct_pensioners'], data['revenue_per_capita'])

#Гипотеза 2: доля пенсионеров vs выручка на человека c помощью корреляции Спирмена
print(f'rho = {corr:.3f}, p = {p_corr:.6f}')
print('Вывод: ' + ('связь значима' if p_corr < 0.05 else 'связь не значима'))

fig, ax = plt.subplots(figsize=(11, 6))
ax.scatter(data['pct_pensioners'], data['revenue_per_capita'], alpha=0.35, s=15, c='#4C9AFF')
z = np.polyfit(data['pct_pensioners'], data['revenue_per_capita'], 1)
xs = np.linspace(data['pct_pensioners'].min(), data['pct_pensioners'].max(), 100)
ax.plot(xs, z[0] * xs + z[1], 'r-', lw=2, label=f'Линейный тренд (rho={corr:.2f})')
ax.set_yscale('log')
ax.set_xlabel('% пенсионеров'); 
ax.set_ylabel('Выручка на жителя')
ax.set_title('Связь доли пенсионеров и выручки на человека'); 
ax.legend()
plt.savefig('hyp_pens.png', dpi=110, bbox_inches='tight')
plt.show()

corr_s, p_s = stats.spearmanr(data['n_shops_nearby'], data['avg_revenue'])

#Гипотеза 3: число магазинов в 3 км vs выручка  c помощью корреляции Спирмена
print(f'rho = {corr_s:.3f}, p = {p_s:.6f}')
print('Вывод: ' + ('связь значима' if p_s < 0.05 else 'связь не значима'))

groups = pd.cut(data['n_shops_nearby'], bins=[-1, 0, 1, 2, 5, 100],
                labels=['0', '1', '2', '3-5', '6+'])
group_rev = data.groupby(groups, observed=True)['avg_revenue'].agg(['mean', 'count']).round(0)
print(group_rev)

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(group_rev.index.astype(str), group_rev['mean'], color='#FF6B6B')
ax.set_xlabel('Магазинов в радиусе 3 км');
ax.set_ylabel('Средняя выручка')
ax.set_title('Выручка автолавки по числу магазинов рядом')
plt.savefig('hyp_shops.png', dpi=110, bbox_inches='tight')
plt.show()
