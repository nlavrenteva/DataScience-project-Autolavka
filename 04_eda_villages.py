print('Cтатистика по деревням:')
villages[['population', 'pct_pensioners', 'avg_income', 'dist_to_city_km']].describe().round(1)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle('Распределения ключевых характеристик деревень', fontsize=14, fontweight='bold')

axes[0, 0].hist(villages['population'], bins=50, color='#4C9AFF', edgecolor='white')
axes[0, 0].set_xscale('log')
axes[0, 0].set_xlabel('Население (log)'); axes[0, 0].set_ylabel('Деревень')
axes[0, 0].set_title(f'Население: медиана = {villages["population"].median():.0f}')

axes[0, 1].hist(villages['pct_pensioners'], bins=30, color='#FF8B94', edgecolor='white')
axes[0, 1].axvline(villages['pct_pensioners'].mean(), color='red', linestyle='--',
                   label=f'Среднее: {villages["pct_pensioners"].mean():.1f}%')
axes[0, 1].set_xlabel('% пенсионеров'); axes[0, 1].set_ylabel('Деревень')
axes[0, 1].set_title('Доля пенсионеров'); axes[0, 1].legend()

axes[1, 0].hist(villages['avg_income'], bins=40, color='#88D498', edgecolor='white')
axes[1, 0].axvline(villages['avg_income'].mean(), color='red', linestyle='--',
                   label=f'Среднее: {villages["avg_income"].mean():.0f} ₽')
axes[1, 0].set_xlabel('Средний доход'); axes[1, 0].set_ylabel('Деревень')
axes[1, 0].set_title('Средний доход на жителя'); axes[1, 0].legend()

axes[1, 1].hist(villages['dist_to_city_km'], bins=40, color='#F5A623', edgecolor='white')
axes[1, 1].set_xlabel('Расстояние от базы), км'); axes[1, 1].set_ylabel('Деревень')
axes[1, 1].set_title('Удалённость от базы')

plt.tight_layout()
plt.savefig('eda_distributions.png', dpi=110, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(11, 6))
sc = ax.scatter(villages['population'], villages['pct_pensioners'],
                c=villages['avg_income'], cmap='viridis', alpha=0.5, s=12)
ax.set_xscale('log')
ax.set_xlabel('Население (log)'); ax.set_ylabel('% пенсионеров')
ax.set_title('Чем меньше деревня - тем больше пенсионеров')
plt.colorbar(sc, ax=ax, label='Средний доход, ₽')
plt.savefig('eda_pensioners_vs_pop.png', dpi=110, bbox_inches='tight')
plt.show()
