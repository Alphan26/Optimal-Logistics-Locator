import time
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from statsmodels.tsa.arima.model import ARIMA

# reading datasets
biomass_history_data = pd.read_csv("dataset/Biomass_History.csv")

# lets see the first 20 data of the dataset to investigate

#print(biomass_history_data.head(20))
# print(distance_matrix_df.head(20))
#print(biomass_history_data.info())
# We have values starting from 2010 until 2017 as well as index, latitude and longitude
# We have index numbers which specifies the location index of harvesting sites.
# For instance, index 1's latitude is 24.668 and longitude is 71.331
#print(biomass_history_data.describe())

# preprocessing step

#print(biomass_history_data.isna().sum())


# There are no null values

# outlier detection
# mean_2010 = biomass_history_data["2010"].mean()
# std_2010 = biomass_history_data["2010"].std()
#
# biomass_history_data["z_score"] = (biomass_history_data["2010"] - mean_2010) / std_2010
# z_score_threshold = 3
#
# outliers_2010 = biomass_history_data[abs(biomass_history_data["z_score"]) > z_score_threshold]
# print(outliers_2010) # 2010 yılındaki verilerde outlier içerenleri yazdırıyoruz.


# def detect_outliers(df, columns, z_score_thr):
#     outlier = pd.DataFrame()
#
#     for column in columns:
#         mean = df[column].mean()
#         std = df[column].std()
#         df[column + "z_score"] = (df[column] - mean) / std
#
#         column_outliers = df[abs(df[column]) > z_score_threshold]
#         outlier = pd.concat([outlier, column_outliers], axis=0)
#
#     return outlier


def detect_outliers(df, columns, z_score_threshold):
    outliers = pd.DataFrame()

    for column in columns:
        mean = df[column].mean()
        std_dev = df[column].std()
        df[column + '_z_score'] = (df[column] - mean) / std_dev

        column_outliers = df[abs(df[column + '_z_score']) > z_score_threshold]
        outliers = pd.concat([outliers, column_outliers], axis=0)

    return outliers


columns_to_check = ["2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017"]
z_score_threshold = 3
outliers = detect_outliers(biomass_history_data, columns_to_check, z_score_threshold)
#print(outliers)
# 269 rows include outliers. Now what to do ?


# now lets investigate the distribution of the biomass avaibility

# year_wise_biomass_avaibility_row_1 = biomass_history_data.loc[1, columns_to_check]
# plt.bar(columns_to_check, year_wise_biomass_avaibility_row_1,color = "blue")
# plt.title("Histogram of Index 1's biomass avaibility ")
# plt.xlabel('Year')
# plt.ylabel('Values')
#
# # Grafik gösterme
# plt.show()

# x ekseninde latitude y ekseninde longitude ve değerler de biomass avaibility olsun istiyorum. (Dağılımı incelemek adına)
# Bu şekilde 7 tane grafik olmalı.



# Create a custom colormap
colors = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]  # Blue, Green, Red
custom_cmap = LinearSegmentedColormap.from_list('custom_colormap', colors, N=256)
years = [2010,2011,2012,2013,2014,2015,2016,2017]

# We draw the scatter plot of the distribution of the biomass avaibilities due to latitude and longitude
# for year in years:
#
# # Scatter plot
#     plt.figure(figsize=(10, 8))
#     plt.scatter(biomass_history_data['Longitude'], biomass_history_data['Latitude'], c=biomass_history_data['2010'], cmap=custom_cmap, s=50)
#     plt.colorbar(label='Biomass')
#
#     plt.title(f'Biomass Distribution by Latitude and Longitude - Year {year}')
#     plt.xlabel('Longitude')
#     plt.ylabel('Latitude')
#
#     plt.show()

# Now lets use some time series algorithms to forecast the amount of biomass avaibility of 2018 and 2019

# There're different approaches that could go. LSTM, ARIMA , Seasonal ARIMA, XGBoost etc.
# LSTM probably won't give decent answers. That's because we have limited amount of data.

# ARIMA modelini eğitme
forecast_steps = 1
forecasts = {}


# sadece birinci satır için gerekli output üretilmeli, ikinci output ayrı gibi gibi
latitude = biomass_history_data['Latitude']
longitude = biomass_history_data['Longitude']

# Tahmin sonuçlarını tutmak için bir liste oluşturun
forecasts = []

# Her bir satır için tahmin yapın ve tahmin sonuçlarını toplayın
for i in range(len(biomass_history_data)):
    timeseries = biomass_history_data.iloc[i, 3:11]  # Biyokütle verileri
    model = ARIMA(timeseries, order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=1)
    #print(type(forecast))
    # print(forecast.values[0])
    # time.sleep(0.001)
    forecasts.append(forecast[0])

print("İşlem tamamlandı")
print(forecasts)
# Scatter plot çizin
plt.figure(figsize=(10, 6))
plt.scatter(longitude, latitude, c=forecasts, cmap='coolwarm', marker='o', s=100)
plt.colorbar(label='Tahmin Edilen Biyokütle Atığı')
plt.xlabel('Boylam')
plt.ylabel('Enlem')
plt.title('Tahmin Edilen Biyokütle Atığı Scatter Plot')
plt.show()

# # Histogram çizimi
# plt.bar(keys, values)
# plt.xlabel('Year')
# plt.ylabel('Biomass Avaibility')
# plt.title('Distribution of Biomass Avaibility Due to Years')
# plt.show()

# Tahmin sonuçlarını görselleştirme
# plt.figure(figsize=(10, 6))
# plt.plot(biomass_history_data.index, biomass_history_data['Biomass'], label='Veriler')
# plt.plot([2018], [forecast[0]], marker='o', color='red', label='2018 Tahmini')
# plt.fill_between([2018], [conf_int[0][0]], [conf_int[0][1]], color='red', alpha=0.2)
# plt.xlabel('Year')
# plt.ylabel('Biomass')
# plt.title('Biomass Prediction for 2018')
# plt.legend()
# plt.show()
