import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle

print("Step 1: Data Collection (Creating CSV data)...")
np.random.seed(42)
dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
sales = 200 + ((dates.year - 2020) * 50) + (np.sin(dates.month * (2 * np.pi / 12)) * 50) + np.random.normal(0, 20, len(dates))

df = pd.DataFrame({'Date': dates, 'Sales': sales})

# Introducing fake missing values and duplicates to strictly meet the project requirements
df.loc[5, 'Sales'] = np.nan 
df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

df.to_csv('raw_sales_data.csv', index=False)
df = pd.read_csv('raw_sales_data.csv')

print("Step 2: Data Cleaning...")
# Convert date column properly
df['Date'] = pd.to_datetime(df['Date'])
# Handle missing values
df = df.dropna()
# Remove duplicates
df = df.drop_duplicates()
df = df.sort_values(by='Date').reset_index(drop=True)

print("Step 4: Feature Engineering...")
# Extract Month, Year, Day
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
# Create time-based features
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Is_Weekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

df.to_csv('cleaned_sales_data.csv', index=False)
print("Data cleaned and saved to cleaned_sales_data.csv")

print("Step 5: Model Building...")
X = df[['Year', 'Month', 'Day', 'DayOfWeek', 'Is_Weekend']]
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

# Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Step 6: Prediction & Step 7: Evaluation...")
predictions = model.predict(X_test)

# Compare actual vs predicted values (saving for Streamlit chart)
results_df = pd.DataFrame({'Date': df['Date'].iloc[y_test.index], 'Actual': y_test, 'Predicted': predictions})
results_df.to_csv('actual_vs_predicted.csv', index=False)

# Evaluation Metrics
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print(f"MAE (Mean Absolute Error): {mae:.2f}")
print(f"RMSE (Root Mean Square Error): {rmse:.2f}")

with open('sales_model.pkl', 'wb') as file:
    pickle.dump(model, file)
print("Model saved successfully.")