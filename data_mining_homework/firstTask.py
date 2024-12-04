# Importing Required Libraries
# I installed the necessary libraries.

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 1. Loading the Dataset
# This dataset includes features like age, chest pain, cholesterol etc.
data = 'data/heart.csv'
heart_data = pd.read_csv(data)

# 2. Preparing the Data
# I simplified the target values. 1 means heart disease, 0 is not.
heart_data['target'] = heart_data['target'].map({1: 1, 2: 0})

# Splitting features and target. x represents the features and y represents the target. y will use for classification model.
X = heart_data.drop(columns=['target'])
y = heart_data['target']

#3. Dividing the Data Set into Training and Testing Sets
# I split the dataset here (%70 training data - %30 test data).
# I selected number 58 for gaining the same outputs.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=58)

#4. Training Decision Tree Classifier
# Creating the Decision Tree Classifier model. I selected number 58 for gaining the same outputs.
dt_model = DecisionTreeClassifier(random_state=58)
# Training the Decision Tree Classifier model.
dt_model.fit(X_train, y_train)

# Predicting using the Decision Tree model
y_pred_dt = dt_model.predict(X_test)

# Calculating performance metrics for Decision Tree
dt_accuracy = accuracy_score(y_test, y_pred_dt)
dt_precision = precision_score(y_test, y_pred_dt)
dt_recall = recall_score(y_test, y_pred_dt)
dt_f1 = f1_score(y_test, y_pred_dt)

#5. Training Random Forest Classifier
# Creating the Random Forest Classifier model. I selected number 58 for gaining the same outputs.
rf_model = RandomForestClassifier(random_state=58)
# Training the Random Forest Classifier model.
rf_model.fit(X_train, y_train)

# Predicting using the Random Forest model
y_pred_rf = rf_model.predict(X_test)

# Calculating performance metrics for Random Forest
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_precision = precision_score(y_test, y_pred_rf)
rf_recall = recall_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf)

#6. Visualizing Predictions
# Visualizing Confusion Matrix for Decision Tree
cm_dt = confusion_matrix(y_test, y_pred_dt)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Visualizing Confusion Matrix for Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Decision Tree Visualization
plt.figure(figsize=(20, 10))
plot_tree(dt_model, filled=True, feature_names=X.columns, class_names=["No Heart Disease", "Heart Disease"])
plt.title("Decision Tree Visualization")
plt.show()

# We Select First Tree in Random Forest
first_tree = rf_model.estimators_[0]

# Random Forest First Decision Tree Visualization
plt.figure(figsize=(15, 10))
plot_tree(first_tree, feature_names=X.columns, class_names=["No Heart Disease", "Heart Disease"], filled=True)
plt.title("Random Forest - First Decision Tree Visualization")
plt.show()

#7. Summarizing Results
results = {
    "Model": ["Decision Tree", "Random Forest"],
    "Accuracy": [dt_accuracy, rf_accuracy],
    "Precision": [dt_precision, rf_precision],
    "Recall": [dt_recall, rf_recall],
    "F1 Score": [dt_f1, rf_f1]
}

results_df = pd.DataFrame(results)
# Showing the results to the users.
print(results_df)