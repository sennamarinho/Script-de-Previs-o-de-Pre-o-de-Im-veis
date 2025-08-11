import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# --- 1. Load Data ---
# Load the datasets
try:
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    data_description = open('data_description.txt', 'r').read() # For reference, not directly used in script logic
    sample_submission = pd.read_csv('sample_submission.csv')
    print("Datasets loaded successfully.")
except FileNotFoundError:
    print("Ensure 'train.csv', 'test.csv', 'data_description.txt', and 'sample_submission.csv' are in the same directory.")
    exit()

# Save original test IDs for submission
test_ids = test_df['Id']

# Drop the 'Id' column from both datasets as it's not a feature
train_df = train_df.drop('Id', axis=1)
test_df = test_df.drop('Id', axis=1)

# --- 2. Feature Engineering and Preprocessing ---

# Combine train and test for consistent preprocessing
all_data = pd.concat((train_df.loc[:,'MSSubClass':'SaleCondition'],
                      test_df.loc[:,'MSSubClass':'SaleCondition']))

print(f"Combined data shape: {all_data.shape}")

# Log transform the target variable 'SalePrice' for the training data
# This is crucial because the evaluation metric is RMSE on the log of the price.
train_df["SalePrice"] = np.log1p(train_df["SalePrice"])

# Handle missing values
# Numerical missing values: Impute with median
for col in ('LotFrontage', 'MasVnrArea', 'GarageYrBlt', 'GarageArea', 'GarageCars',
            'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    if all_data[col].isnull().any():
        all_data[col] = all_data[col].fillna(all_data[col].median())

# Categorical missing values: Fill with 'None' or 'No' depending on context
for col in ('Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
            'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC',
            'Fence', 'MiscFeature', 'MasVnrType', 'MSZoning', 'Utilities', 'Exterior1st',
            'Exterior2nd', 'KitchenQual', 'Electrical', 'Functional', 'SaleType'):
    if all_data[col].isnull().any():
        all_data[col] = all_data[col].fillna('None')

# Specific cases for missing values based on data description
# 'Utilities' has very few non-AllPub values, mostly missing in test set. Drop for simplicity.
if 'Utilities' in all_data.columns:
    all_data = all_data.drop('Utilities', axis=1)

# 'OverallQual' and 'OverallCond' are already numerical. No missing values for these.

# Feature Engineering: Creating new features
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['TotalBath'] = (all_data['FullBath'] + (0.5 * all_data['HalfBath']) +
                         all_data['BsmtFullBath'] + (0.5 * all_data['BsmtHalfBath']))
all_data['YearBuilt-Remod'] = all_data['YearRemodAdd'] - all_data['YearBuilt']
all_data['TotalPorchSF'] = (all_data['OpenPorchSF'] + all_data['EnclosedPorch'] +
                            all_data['3SsnPorch'] + all_data['ScreenPorch'])
all_data['HasPool'] = all_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
all_data['HasGarage'] = all_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
all_data['HasBsmt'] = all_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
all_data['HasFireplace'] = all_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)


# Convert some numerical features into categorical, as their values represent categories
all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)

# Label Encoding for ordinal features where order matters (based on data_description)
# For simplicity, we'll use one-hot encoding for all categoricals below,
# but for better performance, specific ordinal encoding could be applied.

# One-Hot Encode Categorical Features
all_data = pd.get_dummies(all_data)
print(f"Data shape after one-hot encoding: {all_data.shape}")

# Separate back into training and testing sets
X_train = all_data[:len(train_df)]
X_test = all_data[len(train_df):]
y_train = train_df['SalePrice']

# Ensure columns match between training and test sets
# This handles cases where some categories might be present in train but not test, or vice-versa
train_cols = X_train.columns
test_cols = X_test.columns

missing_in_test = set(train_cols) - set(test_cols)
for c in missing_in_test:
    X_test[c] = 0

missing_in_train = set(test_cols) - set(train_cols)
for c in missing_in_train:
    X_train[c] = 0

X_test = X_test[train_cols] # Align columns


# --- 3. Model Training ---

# Random Forest Regressor
print("\nTraining Random Forest Regressor...")
rf_model = RandomForestRegressor(n_estimators=1000, random_state=42, n_jobs=-1, max_features=0.75, min_samples_leaf=1)
rf_model.fit(X_train, y_train)
print("Random Forest training complete.")

# XGBoost Regressor
print("Training XGBoost Regressor...")
xgbr_model = xgb.XGBRegressor(objective='reg:squarederror',
                              n_estimators=2000,
                              learning_rate=0.01,
                              max_depth=5,
                              min_child_weight=1,
                              gamma=0,
                              subsample=0.7,
                              colsample_bytree=0.7,
                              reg_alpha=0.005,
                              random_state=42,
                              n_jobs=-1)
xgbr_model.fit(X_train, y_train)
print("XGBoost training complete.")

# --- 4. Make Predictions ---

# Get predictions from both models
rf_predictions = rf_model.predict(X_test)
xgbr_predictions = xgbr_model.predict(X_test)

# Simple Ensemble: Average the predictions
# You could also use weighted averaging or stacking for better results
ensemble_predictions = (rf_predictions + xgbr_predictions) / 2

# Inverse transform the predictions from log scale back to original scale
final_predictions = np.expm1(ensemble_predictions)

# Ensure no negative predictions (prices cannot be negative)
final_predictions[final_predictions < 0] = 0


# --- 5. Create Submission File ---
submission_df = pd.DataFrame({'Id': test_ids, 'SalePrice': final_predictions})
submission_df.to_csv('submission.csv', index=False)

print("\nSubmission file 'submission.csv' created successfully! 🎉")
print(submission_df.head())

