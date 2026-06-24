import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def generate_synthetic_data():
    print("API baglantisi basarisiz veya zaman asimina ugradi. Sentetik veri uretiliyor...")
    np.random.seed(42)
    countries = [f'Country_{i}' for i in range(1, 51)]
    years = list(range(1990, 2023))
    
    panel_data = []
    for country in countries:
        # Base values for the country to create some panel structure
        country_base_co2 = np.random.uniform(2, 8)
        
        for year in years:
            gdp_growth = np.random.normal(3, 2.5)
            trade_openness = np.random.uniform(20, 100)
            co2_emissions_pc = np.random.normal(5, 1.5) # As requested N(5, 1.5)
            real_interest_rate = np.random.normal(2, 2)
            
            panel_data.append({
                'Country': country,
                'Year': year,
                'GDP_Growth': gdp_growth,
                'Trade_Openness': trade_openness,
                'CO2_Emissions_PC': co2_emissions_pc,
                'Real_Interest_Rate': real_interest_rate,
                'Inflation_Shock': np.random.normal(0, 1.5)
            })
            
    df = pd.DataFrame(panel_data)
    
    # Generate Inflation_Rate iteratively for AR(1)
    df.sort_values(['Country', 'Year'], inplace=True)
    df['Inflation_Rate'] = 0.0
    
    for country in countries:
        mask = df['Country'] == country
        idx = df[mask].index
        
        inf_prev = np.random.uniform(2, 8)
        for i in idx:
            # inf_t = 0.6 * inf_t-1 + 0.8 * CO2 - 0.2 * GDP + 0.05 * Trade - 0.3 * RIR + shock
            inf_t = (0.6 * inf_prev + 
                     0.8 * df.loc[i, 'CO2_Emissions_PC'] - 
                     0.2 * df.loc[i, 'GDP_Growth'] + 
                     0.05 * df.loc[i, 'Trade_Openness'] - 
                     0.3 * df.loc[i, 'Real_Interest_Rate'] + 
                     df.loc[i, 'Inflation_Shock'])
            df.loc[i, 'Inflation_Rate'] = inf_t
            inf_prev = inf_t
            
    # Create Lag1
    df['Inflation_Lag1'] = df.groupby('Country')['Inflation_Rate'].shift(1)
    
    return df

def get_data():
    try:
        import pandas_datareader.data as web
        import datetime
        print("World Bank API'den veri cekilmeye calisiliyor...")
        
        # Test fetch
        test = web.DataReader('NY.GDP.MKTP.KD.ZG', 'wb', start=2020, end=2020)
        
        # Actual fetch
        start_date = datetime.datetime(1990, 1, 1)
        end_date = datetime.datetime(2022, 1, 1)
        
        indicators = {
            'FP.CPI.TOTL.ZG': 'Inflation_Rate',
            'EN.ATM.CO2E.PC': 'CO2_Emissions_PC',
            'NY.GDP.MKTP.KD.ZG': 'GDP_Growth',
            'NE.TRD.GNFS.ZS': 'Trade_Openness',
            'FR.INR.RINR': 'Real_Interest_Rate'
        }
        
        df = web.DataReader(list(indicators.keys()), 'wb', start=start_date, end=end_date)
        df.rename(columns=indicators, inplace=True)
        df.reset_index(inplace=True)
        df.rename(columns={'country': 'Country', 'year': 'Year'}, inplace=True)
        df['Year'] = df['Year'].astype(int)
        
        df.sort_values(['Country', 'Year'], inplace=True)
        df['Inflation_Lag1'] = df.groupby('Country')['Inflation_Rate'].shift(1)
        
        print("World Bank verisi basariyla cekildi.")
        return df
    except Exception as e:
        print(f"API Hatasi: {e}")
        return generate_synthetic_data()

if __name__ == "__main__":
    df = get_data()
    
    cols = ['Inflation_Rate', 'Inflation_Lag1', 'CO2_Emissions_PC', 'GDP_Growth', 'Trade_Openness', 'Real_Interest_Rate']
    # Kural: Yalnızca analize girecek sütunların tamamı NaN olan satırları sil
    df = df.dropna(subset=cols, how='all')
    
    file_name = 'real_greenflation_panel_data.csv'
    df.to_csv(file_name, index=False)
    print(f"\nVeri seti '{file_name}' adli dosyaya kaydedildi. Boyut: {df.shape}")
    
    print("\n--- TANIMLAYICI ISTATISTIKLER ---")
    print(df[cols].describe().round(3))
    
    print("\n--- KORELASYON MATRISI ---")
    print(df[cols].corr().round(3))
