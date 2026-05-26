FEATURES_CLS = ['population', 'pct_pensioners', 'avg_income', 'dist_to_city_km', 'n_shops_nearby', 'nearest_shop_km', 'has_supermarket_nearby']
X = data[FEATURES_CLS]
y = data['is_profitable']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)
print(f'Train: {len(X_train)}, Test: {len(X_test)}')
print(f'Доля прибыльных в train: {y_train.mean():.3f}, test: {y_test.mean():.3f}')

dummy = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
print(f'Baseline (самый частый класс): Accuracy = {accuracy_score(y_test, dummy.predict(X_test)):.3f}')

scaler = StandardScaler()
X_train_sc, X_test_sc = scaler.fit_transform(X_train), scaler.transform(X_test)
logreg = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE).fit(X_train_sc, y_train)
y_pred_lr = logreg.predict(X_test_sc)
y_proba_lr = logreg.predict_proba(X_test_sc)[:, 1]
print(f'\nLogistic Regression: Accuracy = {accuracy_score(y_test, y_pred_lr):.3f}, '
      f'F1 = {f1_score(y_test, y_pred_lr):.3f}, ROC-AUC = {roc_auc_score(y_test, y_proba_lr):.3f}')

rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=20, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]
print(f'Accuracy = {accuracy_score(y_test, y_pred_rf):.3f}, '
      f'F1 = {f1_score(y_test, y_pred_rf):.3f}, ROC-AUC = {roc_auc_score(y_test, y_proba_rf):.3f}')
print()
print(classification_report(y_test, y_pred_rf, target_names=['Не ехать', 'Ехать']))

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Не ехать', 'Ехать'], yticklabels=['Не ехать', 'Ехать'])
axes[0].set_title('Confusion Matrix для Random Forest')
axes[0].set_xlabel('Предсказание'); axes[0].set_ylabel('Истина')

for proba, name in [(y_proba_lr, 'LogReg'), (y_proba_rf, 'RForest')]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    axes[1].plot(fpr, tpr, lw=2, label=f'{name} (AUC={roc_auc_score(y_test, proba):.3f})')
axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.4)
axes[1].set_xlabel('False Positive Rate'); axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC-кривые'); axes[1].legend()

imp = pd.Series(rf.feature_importances_, index=FEATURES_CLS).sort_values()
axes[2].barh(imp.index, imp.values, color='#4C9AFF')
axes[2].set_title('Важность признаков (Random Forest)')

plt.tight_layout()
plt.savefig('cls_results.png', dpi=110, bbox_inches='tight')
plt.show()

rf_full = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=20,
                                 class_weight='balanced', random_state=RANDOM_STATE,
                                 n_jobs=-1).fit(X, y)
data['ml_include_proba'] = rf_full.predict_proba(X)[:, 1].round(3)

THRESHOLD = 0.70
data['ml_include_pred'] = (data['ml_include_proba'] >= THRESHOLD).astype(int)
print(f'Порог включения в пул: {THRESHOLD}')
print(f'Деревень в потенциальном пуле: {data["ml_include_pred"].sum()} из {len(data)}')

(data[['village', 'population', 'avg_income', 'dist_to_city_km',
       'is_profitable', 'ml_include_pred', 'ml_include_proba']]
 .sort_values('ml_include_proba', ascending=False)
 .to_csv('classification_results.csv', index=False))
