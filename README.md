# Airline Passenger Analysis

This project performs a comprehensive analysis of the **Airline Passengers** dataset using time series techniques. The primary goal is to uncover trends, seasonal patterns, and forecast future passenger numbers, which can aid in strategic planning for airlines.

## Table of Contents

- [Problem Statement](#problem-statement)
- [Data](#data)
- [Objective](#objective)
- [Features](#features)
- [Analysis Insights](#analysis-insights)
- [Installation](#installation)
- [Conclusion](#conclusion)

## Problem Statement:

Airlines need to forecast the number of passengers to plan their operations efficiently. Accurate forecasts can help airlines manage their resources, such as aircraft, crew, and ground services, and ensure they can meet customer demand without overcommitting resources, which can lead to unnecessary costs.

## Data:

The dataset used for this analysis contains monthly passenger numbers from January 1949 to December 1960. This time series data captures the seasonal patterns and trends in passenger numbers over the years.

## Objective:

To develop a predictive model that accurately forecasts the number of airline passengers for the next 48 months based on historical data. This analysis aims to help airline companies in capacity planning, resource allocation, and strategic decision-making.


## Features

- **Data Visualization**: Visual representations of the original time series and its decomposed components (trend, seasonality, and residuals).
- **Stationarity Testing**: Employs the Augmented Dickey-Fuller test to check for stationarity and applies differencing where necessary.
- **ARIMA Modeling**: Builds and fits an ARIMA model for forecasting future passenger traffic based on historical data.
- **Forecasting**: Generates forecasts for the next 48 months, visualizing them alongside historical data.

## Analysis Insights

1. **Data Loading and Exploration:**
   - **Loading the Dataset:** The dataset used is (AirlinePassengers.csv)[https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv].
   - **Plotting:** Plotted the time series to visualize the historical trends and patterns in the data.
   - **Observation:** The plot shows an upward trend with seasonal fluctuations.
<br>

   ```python
   url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
   data = pd.read_csv(url, parse_dates=['Month'], index_col='Month')
   data.columns = ['Passengers']
   
   plt.figure(figsize=(12, 6))
   plt.plot(data, label='Monthly Passengers')
   plt.title('Monthly Airline Passengers')
   plt.xlabel('Date')
   plt.ylabel('Number of Passengers')
   plt.legend()
   plt.show()
   ```

2. **Time Series Decomposition:**
   - **Decomposition:** Decompose the time series into trend, seasonality, and residual components using the `seasonal_decompose` method.
   - **Observation:** By visualizing these components, we can understand the underlying patterns in the data.
<br>

   ```python
   decomposition = seasonal_decompose(data, model='multiplicative')
   trend = decomposition.trend
   seasonal = decomposition.seasonal
   residual = decomposition.resid

   plt.figure(figsize=(12, 8))
   plt.subplot(411)
   plt.plot(data, label='Original')
   plt.legend(loc='best')
   plt.subplot(412)
   plt.plot(trend, label='Trend')
   plt.legend(loc='best')
   plt.subplot(413)
   plt.plot(seasonal, label='Seasonality')
   plt.legend(loc='best')
   plt.subplot(414)
   plt.plot(residual, label='Residuals')
   plt.legend(loc='best')
   plt.tight_layout()
   plt.show()
   ```

3. **Stationarity Check:**
   - **ADF Test:** Perform the Augmented Dickey-Fuller (ADF) test on the original series to check for stationarity. A non-stationary series has a unit root, showing trends or seasonality that must be removed.
   - **Differencing:** Apply differencing to the series if it is non-stationary and perform the ADF test again.
<br>

   ```python
   from statsmodels.tsa.stattools import adfuller

   def adf_test(series):
       result = adfuller(series)
       print('ADF Statistic: %f' % result[0])
       print('p-value: %f' % result[1])
       print('Critical Values:')
       for key, value in result[4].items():
           print('\t%s: %.3f' % (key, value))

   print('ADF Test on Original Series:')
   adf_test(data['Passengers'])

   data_diff = data.diff().dropna()
   print('ADF Test on Differenced Series:')
   adf_test(data_diff['Passengers'])
   ```

4. **Model Selection:**
   - **ACF and PACF Plots:** Plot the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) to help determine the order of the ARIMA model.
   - **ARIMA Model:** Fit an ARIMA model to the data based on insights from the ACF and PACF plots.
<br>

   ```python
   from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

   plt.figure(figsize=(12, 6))
   plt.subplot(121)
   plot_acf(data_diff, ax=plt.gca())
   plt.subplot(122)
   plot_pacf(data_diff, ax=plt.gca())
   plt.show()

   from statsmodels.tsa.arima.model import ARIMA
   model = ARIMA(data, order=(4, 1, 4))
   model_fit = model.fit()
   print(model_fit.summary())
   ```

5. **Model Diagnostics:**
   - **Residual Analysis:** Analyze the residuals of the fitted model to check for randomness and normality.
   - **Density Plot:** Plot the density of the residuals to ensure they follow a normal distribution.
<br>

   ```python
   residuals = model_fit.resid
   plt.figure(figsize=(12, 6))
   plt.plot(residuals)
   plt.title('Residuals')
   plt.show()

   sns.histplot(residuals, kde=True)
   plt.title('Density Plot of Residuals')
   plt.show()
   ```

6. **Forecasting:**
   - **Forecasting with ARIMA:** Forecast the number of passengers for the next 48 months using the fitted ARIMA model.
   - **Plot Forecast:** Visualize the forecast alongside the historical data to assess the model's performance.
<br>

   ```python
   forecast_steps = 48
   forecast = model_fit.forecast(steps=forecast_steps)
   forecast_index = pd.date_range(data.index[-1], periods=forecast_steps, freq='MS')

   forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=['Forecast'])

   plt.figure(figsize=(12, 6))
   plt.plot(data, label='Historical Data')
   plt.plot(forecast, label='Forecast', color='red')
   plt.title('Forecast of Monthly Passengers')
   plt.xlabel('Date')
   plt.ylabel('Number of Passengers')
   plt.legend()
   plt.show()
   ```

## Installation

To set up the project locally, follow these steps:

1. **Clone the repository**:<br>
   ```bash
   git clone https://github.com/Diya050/Airline_Passenger_Analysis.git
   cd Airline_Passenger_Analysis
   ```
   
2. **Install the required packages**:<br>
   Ensure you have Python3 installed, then run:<br>
   ```bash
   pip install numpy pandas matplotlib
   ```

3. **Run the Jupyter notebook**:<br>
   ```bash
   jupyter notebook Airline_Passengers.ipynb
   ```

## Conclusion

This analysis effectively demonstrates how time series techniques can be applied to airline passenger data. The ARIMA model successfully identifies underlying trends and seasonal patterns, providing reliable forecasts for future passenger counts. These insights can assist airlines in making data-driven decisions related to capacity planning, scheduling, and marketing strategies. By leveraging historical data, the analysis underscores the importance of forecasting in optimizing operational efficiency and enhancing customer experience.
