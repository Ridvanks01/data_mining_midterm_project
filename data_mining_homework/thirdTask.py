# Importing Required Libraries
# I installed the necessary libraries
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

#1. Loading the California Housing Dataset
# This dataset includes features like income, house ages, number of rooms etc.
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="Price")

#2. Scaling Features
# I created a scaler for scaling the features. This scaler makes the mean 0 and makes the standard deviation 1. This process makes machine learning models work better.
scaler = StandardScaler()
# scaled x feature.
X_scaled = scaler.fit_transform(X)

#3. Dividing the Data Set into Training and Testing Sets
# I split the dataset here (%70 training data - %30 test data).
# I selected number 58 for gaining the same outputs.
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=58)

#4. Linear Regression Model
# Creating the lineer regression model.
lr_model = LinearRegression()
# Training the lineer regression model.
lr_model.fit(X_train, y_train)

# Predictions and Performance Metrics
# Predicting process
y_pred_lr = lr_model.predict(X_test)
print("\nLinear Regression Performance:")
print("MSE:", mean_squared_error(y_test, y_pred_lr))
print("MAE:", mean_absolute_error(y_test, y_pred_lr))
print("R² Score:", r2_score(y_test, y_pred_lr))

#5. Random Forest Regression Model
# Creating the random forest model
rf_model = RandomForestRegressor(random_state=58, n_estimators=100)
# Training the random forest model
rf_model.fit(X_train, y_train)

# Predictions and Performance Metrics
# Predicting process
y_pred_rf = rf_model.predict(X_test)
print("\nRandom Forest Performance:")
print("MSE:", mean_squared_error(y_test, y_pred_rf))
print("MAE:", mean_absolute_error(y_test, y_pred_rf))
print("R² Score:", r2_score(y_test, y_pred_rf))

#6. Visualizing Predictions
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lr, label="Linear Regression", alpha=0.7)
plt.scatter(y_test, y_pred_rf, label="Random Forest", alpha=0.7)
plt.title("Actual vs Predicted Values")
plt.xlabel("Actual Values (House Prices)")
plt.ylabel("Predicted Values")
plt.legend()
plt.show()

# 7. Importance Ranking of Features (Random Forest)
feature_importances = rf_model.feature_importances_
feature_names = X.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=feature_names)
plt.title("Feature Importances from Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()