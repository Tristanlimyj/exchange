import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm

# Load the dataset
data = pd.read_csv('./salaries.csv')

# 1. Data Exploration and Manipulation

# 1.1 Dataset Overview
def dataset_overview(data):
    print("Dataset Head:\n", data.head())
    print("\nDataset Info:\n")
    data.info()
    print("\nDescriptive Statistics:\n", data.describe())
    
dataset_overview(data)

# Identify the target variable 
target_variable = 'salary'
if target_variable not in data.columns:
    raise ValueError(f"Target column '{target_variable}' not found in the dataset.")

# 1.2 Check for Missing Values
missing_values = data.isnull().sum()
print("\nMissing Values:\n", missing_values)

# Handle missing values (Impute with mean for numerical and mode for categorical)
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = data[col].fillna(data[col].mode()[0])
    else:
        data[col] = data[col].fillna(data[col].mean())

# 1.3 Convert Categorical Variables
categorical_columns = data.select_dtypes(include=['object']).columns
print("\nCategorical Columns:\n", categorical_columns)

data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# 1.4 Correlation Analysis
plt.figure(figsize=(10, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# 1.5 Additional Visualization
sns.boxplot(data=data, x=target_variable, y=data.columns[1])  # Replace data.columns[1] with an appropriate feature
plt.title('Box Plot of Target vs Feature')
plt.show()

# 2. Multilinear Regression Model

# 2.1 Preprocessing
X = data.drop(target_variable, axis=1)
y = data[target_variable]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 2.2 Model Creation
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# 2.3 Model Summary
X_train_const = sm.add_constant(X_train)
ols_model = sm.OLS(y_train, X_train_const).fit()
print(ols_model.summary())

# 2.4 Make Predictions
predictions = linear_model.predict(X_test)

# 2.5 Evaluate Model Performance
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print("\nLinear Regression - MAE:", mae)
print("Linear Regression - RMSE:", rmse)


# 3. Tree Regression Model

# 3.1 Model Creation
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)

# 3.2 Feature Importance
feature_importances = tree_model.feature_importances_
important_features = pd.Series(feature_importances, index=X.columns).sort_values(ascending=False)
print("\nFeature Importances:\n", important_features)

# 3.3 Make Predictions
tree_predictions = tree_model.predict(X_test)

# 3.4 Evaluate Model Performance
tree_mae = mean_absolute_error(y_test, tree_predictions)
tree_rmse = np.sqrt(mean_squared_error(y_test, tree_predictions))
print("\nTree Regression - MAE:", tree_mae)
print("Tree Regression - RMSE:", tree_rmse)

# 3.5 Fine-Tuning
# Train model with top features (e.g., top 5 features)
top_features = important_features.head(5).index
X_train_top = X_train[:, [X.columns.get_loc(col) for col in top_features]]
X_test_top = X_test[:, [X.columns.get_loc(col) for col in top_features]]

tree_model.fit(X_train_top, y_train)
tree_predictions_top = tree_model.predict(X_test_top)

tree_mae_top = mean_absolute_error(y_test, tree_predictions_top)
tree_rmse_top = np.sqrt(mean_squared_error(y_test, tree_predictions_top))
print("\nFine-Tuned Tree Regression - MAE:", tree_mae_top)
print("Fine-Tuned Tree Regression - RMSE:", tree_rmse_top)

# 4. Extra Mile: Random Forest Regression Model

# Train the Random Forest model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Make Predictions
rf_predictions = rf_model.predict(X_test)

# Evaluate Model Performance
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
print("\nRandom Forest Regression - MAE:", rf_mae)
print("Random Forest Regression - RMSE:", rf_rmse)
