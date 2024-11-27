# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load the dataset
file_path = r"C:\Users\lenovo\Desktop\Stock Dataset\stocks.csv"  # Replace with your actual path
stocks_data = pd.read_csv(file_path)

# Step 1: Data Inspection
print("Dataset Info:")
print(stocks_data.info())
print("\nSummary Statistics:")
print(stocks_data.describe())

# Step 2: Data Preprocessing
# Handle missing values (if any)
if stocks_data.isnull().sum().sum() > 0:
    stocks_data.fillna(method='ffill', inplace=True)

# Detect outliers using IQR for Volume
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] < lower_bound) | (data[column] > upper_bound)]

outliers_volume = detect_outliers_iqr(stocks_data, 'Volume')
print(f"\nOutliers in Volume: {len(outliers_volume)}")

# Optionally, remove the outliers from Volume
stocks_data = stocks_data[~stocks_data.index.isin(outliers_volume.index)]

# Standardize numeric features
scaler = StandardScaler()
numeric_cols = stocks_data.select_dtypes(include=[np.number])
scaled_numeric_data = scaler.fit_transform(numeric_cols)
scaled_df = pd.DataFrame(scaled_numeric_data, columns=numeric_cols.columns)
print("\nStandardized Numeric Data Sample:")
print(scaled_df.head())

# Encode categorical variables (Ticker column)
encoder = OneHotEncoder()
encoded_ticker = encoder.fit_transform(stocks_data[['Ticker']])
encoded_ticker_df = pd.DataFrame(encoded_ticker.toarray(), columns=encoder.get_feature_names_out(['Ticker']))
stocks_data = pd.concat([stocks_data.reset_index(drop=True), encoded_ticker_df], axis=1)
print("\nEncoded Ticker Columns:")
print(encoded_ticker_df.head())

# Step 3: Data Aggregation and Grouping
aggregated_data = stocks_data.groupby('Ticker').agg({
    'Volume': 'sum',
    'Close': 'mean'
}).rename(columns={'Volume': 'Total Volume', 'Close': 'Average Close Price'})

print("\nAggregated Data:")
print(aggregated_data)

# Step 4: Exploratory Data Analysis (EDA)
# Line plot for closing price trends
plt.figure(figsize=(10, 6))
sns.lineplot(data=stocks_data, x='Date', y='Close', hue='Ticker', legend=False)
plt.title("Stock Closing Price Over Time")
plt.xticks(rotation=45)
plt.show()

# Distribution of Volume
plt.figure(figsize=(10, 6))
sns.histplot(stocks_data['Volume'], bins=30, kde=True, color='blue')
plt.title("Volume Distribution")
plt.show()

# Correlation Heatmap (only numeric)
plt.figure(figsize=(8, 6))
sns.heatmap(scaled_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Step 5: Key Insights
key_insights = {
    "Highest Trading Volume": aggregated_data['Total Volume'].idxmax(),
    "Highest Average Close Price": aggregated_data['Average Close Price'].idxmax(),
    "Lowest Average Close Price": aggregated_data['Average Close Price'].idxmin()
}

print("\nKey Insights:")
for insight, ticker in key_insights.items():
    print(f"{insight}: {ticker}")

# Step 6: Quantitative Analysis
# Calculate daily returns
stocks_data['Daily Return'] = stocks_data['Adj Close'].pct_change()

# Daily Returns Over Time
plt.figure(figsize=(10, 6))
sns.lineplot(data=stocks_data, x='Date', y='Daily Return', hue='Ticker', legend=False)
plt.title("Daily Returns Over Time")
plt.xticks(rotation=45)
plt.show()

# Calculate volatility (standard deviation of daily returns)
volatility = stocks_data.groupby('Ticker')['Daily Return'].std()

# Volatility Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=volatility.index, y=volatility.values)
plt.title("Volatility (Standard Deviation of Daily Returns) by Ticker")
plt.xlabel('Ticker')
plt.ylabel('Volatility (Std of Daily Returns)')
plt.show()

# Cumulative Returns
stocks_data['Cumulative Return'] = (1 + stocks_data['Daily Return']).cumprod()

# Cumulative Returns Over Time
plt.figure(figsize=(10, 6))
sns.lineplot(data=stocks_data, x='Date', y='Cumulative Return', hue='Ticker', legend=False)
plt.title("Cumulative Returns Over Time")
plt.xticks(rotation=45)
plt.show()

# Risk vs Return (Scatter Plot)
mean_returns = stocks_data.groupby('Ticker')['Daily Return'].mean()

# Scatter Plot for Risk vs Return
risk_return = pd.DataFrame({
    'Mean Return': mean_returns,
    'Volatility': volatility
})

plt.figure(figsize=(10, 6))
sns.scatterplot(data=risk_return, x='Volatility', y='Mean Return', hue=risk_return.index)
plt.title("Risk vs Return (Volatility vs Mean Return)")
plt.xlabel('Volatility')
plt.ylabel('Mean Return')
plt.show()

# Box plot for Opening prices
plt.figure(figsize=(10, 6))
sns.boxplot(data=stocks_data, x='Ticker', y='Open')
plt.title("Box Plot of Opening Prices by Ticker")
plt.show()

# Box plot for Closing prices
plt.figure(figsize=(10, 6))
sns.boxplot(data=stocks_data, x='Ticker', y='Close')
plt.title("Box Plot of Closing Prices by Ticker")
plt.show()
