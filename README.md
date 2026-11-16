# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 04-11-2026
### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# 1. Load the dataset
df = pd.read_csv('cardekho (1).csv')

# ---------------------------------------------------------
# 2. ARMA(1,1) Process on 'selling_price' (First 1,000 rows)
# ---------------------------------------------------------
data1 = df['selling_price'].dropna().values[:1000]

# Fit an ARMA(1,1) model -> ARIMA(order=(1, 0, 1))
arma1_model = ARIMA(data1, order=(1, 0, 1)).fit()
print(arma1_model.summary())

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(data1, label='Actual Selling Price')
plt.plot(
    arma1_model.fittedvalues, color='red', alpha=0.7, label='ARMA(1,1) Fitted'
)
plt.title('Selling Price (First 1,000 rows) with ARMA(1,1) Fit')
plt.legend()

plt.subplot(3, 1, 2)
plot_acf(data1, lags=30, ax=plt.gca())
plt.title('Autocorrelation - Selling Price')

plt.subplot(3, 1, 3)
plot_pacf(data1, lags=30, ax=plt.gca())
plt.title('Partial Autocorrelation - Selling Price')

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 3. ARMA(2,2) Process on 'km_driven' (All 8,128 rows)
# ---------------------------------------------------------
data2 = df['km_driven'].dropna().values

# Fit an ARMA(2,2) model -> ARIMA(order=(2, 0, 2))
arma2_model = ARIMA(data2, order=(2, 0, 2)).fit()
print(arma2_model.summary())

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(data2, label='Actual Km Driven')
plt.plot(
    arma2_model.fittedvalues, color='red', alpha=0.7, label='ARMA(2,2) Fitted'
)
plt.title('Km Driven (All Rows) with ARMA(2,2) Fit')
plt.legend()

plt.subplot(3, 1, 2)
plot_acf(data2, lags=30, ax=plt.gca())
plt.title('Autocorrelation - Km Driven')

plt.subplot(3, 1, 3)
plot_pacf(data2, lags=30, ax=plt.gca())
plt.title('Partial Autocorrelation - Km Driven')

plt.tight_layout()
plt.show()

```
### OUTPUT:

SIMULATED ARMA(1,1) PROCESS:

<img width="1112" height="256" alt="image" src="https://github.com/user-attachments/assets/dc5a0476-ddde-4ed5-aa99-753152df7fb1" />


Autocorrelation

<img width="1112" height="256" alt="image" src="https://github.com/user-attachments/assets/744371dd-de6e-4770-ba9f-69e737e5609f" />


Partial Autocorrelation

<img width="1112" height="237" alt="image" src="https://github.com/user-attachments/assets/b4c8bae2-be1b-4a86-9d4e-1bf3ec4df0f1" />


SIMULATED ARMA(2,2) PROCESS:

<img width="1112" height="250" alt="image" src="https://github.com/user-attachments/assets/192caa0d-0ef0-4457-be74-9c0a6ec436cc" />


Partial Autocorrelation

<img width="1112" height="232" alt="image" src="https://github.com/user-attachments/assets/c70065f2-1611-4e9a-93ae-a0c920288c70" />


Autocorrelation
<img width="1108" height="247" alt="image" src="https://github.com/user-attachments/assets/e63031fc-34c3-4c56-8392-88bc094e0e41" />


### RESULT:
Thus, a python program is created to fir ARMA Model successfully.
