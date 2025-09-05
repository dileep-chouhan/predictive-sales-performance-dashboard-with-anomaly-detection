import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import iqr
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
dates = pd.date_range(start='2023-01-01', periods=365)
sales = 1000 + 200 * np.sin(2 * np.pi * np.arange(365) / 365) + np.random.normal(0, 100, 365) # Seasonal trend with noise
sales[100:110] += 500 # Simulate a temporary sales spike
sales[250:260] -= 300 # Simulate a temporary sales dip
df = pd.DataFrame({'Date': dates, 'Sales': sales})
# --- 2. Anomaly Detection ---
# Simple anomaly detection using Interquartile Range (IQR)
Q1 = df['Sales'].quantile(0.25)
Q3 = df['Sales'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df['Anomaly'] = ((df['Sales'] < lower_bound) | (df['Sales'] > upper_bound)).astype(int)
# --- 3. Data Visualization ---
#Sales Trend
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Sales'], label='Sales')
plt.scatter(df[df['Anomaly'] == 1]['Date'], df[df['Anomaly'] == 1]['Sales'], color='red', label='Anomalies')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Sales Trend with Anomaly Detection')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('sales_trend_anomalies.png')
print("Plot saved to sales_trend_anomalies.png")
# Monthly Average Sales
monthly_sales = df.resample('M', on='Date')['Sales'].mean()
plt.figure(figsize=(10, 6))
sns.lineplot(x=monthly_sales.index, y=monthly_sales.values)
plt.title('Average Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Average Sales')
plt.grid(True)
plt.tight_layout()
plt.savefig('monthly_sales_trend.png')
print(f"Plot saved to monthly_sales_trend.png")
#Boxplot to show data distribution and outliers
plt.figure(figsize=(8,6))
sns.boxplot(y=df['Sales'])
plt.title('Sales Data Distribution')
plt.ylabel('Sales')
plt.savefig('sales_distribution.png')
print(f"Plot saved to sales_distribution.png")