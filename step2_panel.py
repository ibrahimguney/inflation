import pandas as pd
from linearmodels.panel import PanelOLS, FirstDifferenceOLS
import statsmodels.api as sm

# Veriyi yukle
df = pd.read_csv('real_greenflation_panel_data.csv')

# Panel verisi analizi oncesi NA degerleri kaldir
df = df.dropna()

# MultiIndex ayarla (Country ve Year)
df = df.set_index(['Country', 'Year'])

# Bagimli ve Bagimsiz Degiskenler
exog_vars = ['Inflation_Lag1', 'CO2_Emissions_PC', 'GDP_Growth', 'Trade_Openness', 'Real_Interest_Rate']
endog_var = 'Inflation_Rate'

exog = df[exog_vars]
exog = sm.add_constant(exog)
endog = df[endog_var]

print("\n--- a) Sabit Etkiler (Fixed Effects - FE) Modeli ---")
mod_fe = PanelOLS(endog, exog, entity_effects=True)
res_fe = mod_fe.fit(cov_type='robust')
print(res_fe.summary.tables[1])
print(f"R-squared: {res_fe.rsquared:.4f}")

print("\n--- b) Birinci Farklar (First Difference - FD) Modeli ---")
mod_fd = FirstDifferenceOLS(endog, df[exog_vars])
res_fd = mod_fd.fit(cov_type='robust')
print(res_fd.summary.tables[1])
print(f"R-squared: {res_fd.rsquared:.4f}")
