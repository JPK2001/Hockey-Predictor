import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load your CSV dataset (replace 'your_dataset.csv' with the actual file path)
data = pd.read_csv('PlayedNHL.csv')

# Extract relevant columns
columns_of_interest = ['Date', 'Visitor', 'Home', 'G', 'Att.', 'LOG']
df = data[columns_of_interest]

# Feature engineering (example: goal difference)
df['Goal_Difference'] = df['G_Home'] - df['G_Visitor']

# One-hot encode team names
df_encoded = pd.get_dummies(df, columns=['Visitor', 'Home'], drop_first=True)

# Split data into training and validation sets
X = df_encoded.drop(['G', 'Att.', 'LOG'], axis=1)
y_goals = df_encoded['G']
y_attendance = df_encoded['Att.']
y_length = df_encoded['LOG']

X_train, X_val, y_goals_train, y_goals_val, y_attendance_train, y_attendance_val, y_length_train, y_length_val = train_test_split(
    X, y_goals, y_attendance, y_length, test_size=0.2, random_state=42
)

# Initialize Random Forest regressor models
goals_model = RandomForestRegressor(n_estimators=100, random_state=42)
attendance_model = RandomForestRegressor(n_estimators=100, random_state=42)
length_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train models
goals_model.fit(X_train, y_goals_train)
attendance_model.fit(X_train, y_attendance_train)
length_model.fit(X_train, y_length_train)

# Make predictions
goals_predictions = goals_model.predict(X_val)
attendance_predictions = attendance_model.predict(X_val)
length_predictions = length_model.predict(X_val)

# Evaluate model performance
goals_mae = mean_absolute_error(y_goals_val, goals_predictions)
attendance_mae = mean_absolute_error(y_attendance_val, attendance_predictions)
length_mae = mean_absolute_error(y_length_val, length_predictions)

print(f"Goals MAE: {goals_mae:.2f}")
print(f"Attendance MAE: {attendance_mae:.2f}")
print(f"Game Length MAE: {length_mae:.2f}")

# Predict outcomes for scheduled games (replace 'scheduled_games.csv' with actual data)
scheduled_games = pd.read_csv('UnplayedNHL2.csv')
scheduled_games_encoded = pd.get_dummies(scheduled_games, columns=['Visitor', 'Home'], drop_first=True)
predictions_goals = goals_model.predict(scheduled_games_encoded)
predictions_attendance = attendance_model.predict(scheduled_games_encoded)
predictions_length = length_model.predict(scheduled_games_encoded)

# Display predictions (you can create visualizations here)
print("Predicted Goals:", predictions_goals)
print("Predicted Attendance:", predictions_attendance)
print("Predicted Game Length:", predictions_length)