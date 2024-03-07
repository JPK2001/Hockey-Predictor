import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

# Load CSV dataset
data = pd.read_csv('PlayedNHL.csv')

# Extract relevant columns
relevant_columns = ['Visitor', 'Home', 'G_VIS', 'G_HOME', 'Att.', 'LOG']
data = data[relevant_columns]

# Check for and handle missing values if any
data.dropna(inplace=True)

# Convert 'LOG' column to minutes
def convert_to_minutes(time_str):
    if isinstance(time_str, str):
        time_components = time_str.split(':')
        return int(time_components[0]) * 60 + int(time_components[1])
    else:
        return None

data['LOG_minutes'] = data['LOG'].apply(convert_to_minutes)

# Drop rows with missing values in 'LOG_minutes'
data.dropna(subset=['LOG_minutes'], inplace=True)

# Explore the dataset
print(data.head())
print(data.describe())

# Visualize distributions and relationships
sns.pairplot(data)
plt.show()

# Create new features
data['Goal_Difference'] = data['G_VIS'] - data['G_HOME']

# Encode categorical variables
data_encoded = pd.get_dummies(data, columns=['Visitor', 'Home'])

# Split dataset into training and validation sets
X = data_encoded.drop(['G_VIS', 'G_HOME', 'Att.', 'LOG', 'LOG_minutes'], axis=1)
y_goals = data_encoded['G_VIS'] - data_encoded['G_HOME'] # Using goal difference as target variable
y_attendance = data_encoded['Att.']
y_length = data_encoded['LOG_minutes']

X_train, X_val, y_train_goals, y_val_goals, y_train_attendance, y_val_attendance, y_train_length, y_val_length = train_test_split(X, y_goals, y_attendance, y_length, test_size=0.2, random_state=42)

# Train Random Forest models
rf_goals = RandomForestRegressor()
rf_goals.fit(X_train, y_train_goals)

rf_attendance = RandomForestRegressor()
rf_attendance.fit(X_train, y_train_attendance)

rf_length = RandomForestRegressor()
rf_length.fit(X_train, y_train_length)

# Evaluate models
pred_goals = rf_goals.predict(X_val)
mae_goals = mean_absolute_error(y_val_goals, pred_goals)
print("Mean Absolute Error for Goals:", mae_goals)

pred_attendance = rf_attendance.predict(X_val)
mae_attendance = mean_absolute_error(y_val_attendance, pred_attendance)
print("Mean Absolute Error for Attendance:", mae_attendance)

pred_length = rf_length.predict(X_val)
mae_length = mean_absolute_error(y_val_length, pred_length)
print("Mean Absolute Error for Game Length:", mae_length)

# Assuming you have a separate dataset for scheduled games
scheduled_data = pd.read_csv('UnplayedNHL.csv')

# Feature engineering for scheduled games
scheduled_data['Goal_Difference'] = scheduled_data['G_HOME'] - scheduled_data['G_VIS']

# Check if 'LOG_minutes' column exists before applying the conversion function
if 'LOG_minutes' in scheduled_data.columns:
    scheduled_data['LOG'] = scheduled_data['LOG_minutes'].apply(convert_to_minutes)

# Ensure consistency of columns with the trained model
scheduled_data_encoded = pd.get_dummies(scheduled_data, columns=['Visitor', 'Home'])

# Ensure consistency of columns with the trained model
missing_cols = set(data_encoded.columns) - set(scheduled_data_encoded.columns)
for col in missing_cols:
    scheduled_data_encoded[col] = 0
missing_cols = set(scheduled_data_encoded.columns) - set(data_encoded.columns)
for col in missing_cols:
    data_encoded[col] = 0

scheduled_data_encoded = scheduled_data_encoded[data_encoded.columns]

# Predict outcomes for scheduled games
scheduled_goals = rf_goals.predict(scheduled_data_encoded)
scheduled_attendance = rf_attendance.predict(scheduled_data_encoded)
scheduled_length = rf_length.predict(scheduled_data_encoded)

# Display predictions
print("Predicted Goals for Scheduled Games:", scheduled_goals)
print("Predicted Attendance for Scheduled Games:", scheduled_attendance)
print("Predicted Game Length for Scheduled Games:", scheduled_length)

# Plotting predictions
plt.bar(range(len(scheduled_data)), scheduled_goals)
plt.xlabel('Game')
plt.ylabel('Predicted Goals')
plt.title('Predicted Goals for Scheduled Games')
plt.show()

plt.plot(range(len(scheduled_data)), scheduled_attendance)
plt.xlabel('Game')
plt.ylabel('Predicted Attendance')
plt.title('Predicted Attendance for Scheduled Games')
plt.show()

plt.hist(scheduled_length)
plt.xlabel('Game Length (minutes)')
plt.ylabel('Frequency')
plt.title('Predicted Game Length Distribution for Scheduled Games')
plt.show()
