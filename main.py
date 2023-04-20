import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Data collection and exploration
train_df = pd.read_csv('train.csv')
print(train_df.head())

# Data cleaning and preprocessing
# Drop columns with high percentage of missing values
train_df = train_df.drop(['Name', 'RescuerID', 'Description', 'PetID'], axis=1)
train_df = train_df.dropna(thresh=0.75*len(train_df), axis=1)

# Convert categorical variables to numerical
train_df = pd.get_dummies(train_df, columns=['Type', 'Breed1', 'Breed2', 'Color1', 'Color2', 'Color3', 'Gender', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'State'])

# Feature engineering and selection
# Split dataset into features and target
X = train_df.drop(['AdoptionSpeed'], axis=1)
y = train_df['AdoptionSpeed']

# Select best K features using F-regression
selector = SelectKBest(f_regression, k=10)
X = selector.fit_transform(X, y)

# Model training and evaluation
# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('MSE:', mse)
