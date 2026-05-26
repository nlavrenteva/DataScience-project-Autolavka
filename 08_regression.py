class DemandModel:
    FEATURES = ['population', 'pct_pensioners', 'avg_income', 'dist_to_city_km',
                'n_shops_nearby', 'nearest_shop_km', 'has_supermarket_nearby']

    def __init__(self):
        self.model = lgb.LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                       num_leaves=15, min_child_samples=10,
                                       random_state=RANDOM_STATE, verbose=-1)

    def cross_validate(self, X, y, n_splits=5):
        y_log = np.log1p(y)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        maes, r2s = [], []
        for tr, te in kf.split(X):
            self.model.fit(X.iloc[tr], y_log.iloc[tr])
            pred = np.clip(np.expm1(self.model.predict(X.iloc[te])), 0, None)
            maes.append(mean_absolute_error(y.iloc[te], pred))
            r2s.append(r2_score(y.iloc[te], pred))
        return np.mean(maes), np.mean(r2s)

    def fit(self, X, y):
        self.model.fit(X, np.log1p(y))
        return self

    def predict(self, X):
        return np.clip(np.expm1(self.model.predict(X)), 0, None)

    def feature_importance(self):
        return pd.Series(self.model.feature_importances_, index=self.FEATURES).sort_values()


X_reg, y_reg = data[DemandModel.FEATURES], data['avg_revenue']
base_mae = -cross_val_score(DummyRegressor(strategy='median'), X_reg, y_reg,
                            cv=5, scoring='neg_mean_absolute_error').mean()
demand = DemandModel()
mae, r2 = demand.cross_validate(X_reg, y_reg)
print(f'Baseline MAE: {base_mae:.0f}')
print(f'LightGBM cross-val MAE: {mae:.0f} ₽, R2= {r2:.3f}')
print(f'Улучшение к baseline: {(1 - mae / base_mae) * 100:.1f}%')

X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_reg, test_size=0.25, random_state=RANDOM_STATE)
demand.fit(X_tr, y_tr)
pred = demand.predict(X_te)
print(f'MAE:{mean_absolute_error(y_te, pred):.0f}')
print(f'RMSE: {np.sqrt(mean_squared_error(y_te, pred)):.0f}')
print(f'R2: {r2_score(y_te, pred):.3f}')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(y_te, pred, alpha=0.4, s=15, color='#4C9AFF')
lim = [max(1, min(y_te.min(), pred.min())), max(y_te.max(), pred.max())]
axes[0].plot(lim, lim, 'r--', lw=2, label='Идеальное предсказание')
axes[0].set_xscale('log'); axes[0].set_yscale('log')
axes[0].set_xlabel('Истинная выручка'); axes[0].set_ylabel('Предсказанная выручка')
axes[0].set_title('Предсказание b симулированная выручка')
axes[0].legend()

demand_full = DemandModel().fit(X_reg, y_reg)
imp = demand_full.feature_importance()
axes[1].barh(imp.index, imp.values, color='#88D498')
axes[1].set_title('Важность признаков (LightGBM)')

plt.tight_layout()
plt.savefig('reg_results.png', dpi=110, bbox_inches='tight')
plt.show()

data['predicted_revenue'] = demand_full.predict(X_reg).round(0)
print('\nТоп-5 деревень по прогнозу выручки:')
print(data.nlargest(5, 'predicted_revenue')[
    ['village', 'population', 'avg_income', 'predicted_revenue']].to_string(index=False))
