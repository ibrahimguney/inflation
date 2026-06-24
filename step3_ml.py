import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('real_greenflation_panel_data.csv')
df = df.dropna()

features = ['Inflation_Lag1', 'CO2_Emissions_PC', 'GDP_Growth', 'Trade_Openness', 'Real_Interest_Rate']
target = 'Inflation_Rate'

X = df[features]
y = df[target]
groups = df['Country']

# Ülke-yıl bütünlüğünü bozmamak için GroupShuffleSplit kullanarak ülkeleri %80-%20 ayırıyoruz.
# Böylece bir ülkenin bazı yılları eğitimde, bazıları testte olmaz.
gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

model = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n--- XGBoost Model Performansi ---")
print(f"Test RMSE: {rmse:.4f}")
print(f"Test R-kare: {r2:.4f}")

print("\n--- XGBoost Feature Importance (Olasiligi/Agirligi) ---")
importance = model.feature_importances_
feature_imp_df = pd.DataFrame({'Feature': features, 'Importance': importance})
feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)
print(feature_imp_df.to_string(index=False))
