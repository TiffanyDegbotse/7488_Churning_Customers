# -*- coding: utf-8 -*-
"""Churning.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DP5xbNOzecTnyJnRq24Oda0Hxb5k_sqR
"""

#Here, I am mounting and loading my file
import pandas as pd
import os
import numpy as np
from google.colab import drive
drive.mount('/content/drive')
file_path= os.path.abspath(r"/content/drive/MyDrive/CustomerChurn_dataset.csv")
# Creating a variable with the name "df" that keeps the data
df= pd.read_csv(file_path)
df

"""EXPLORATORY DATA ANALYSIS"""

# The customerID was removed because it is not really relevant in predicting the churn
remove_columns =['customerID']
df=df.drop(columns=remove_columns)

# Checking for values that have misssing values greater than 30 %
missing_percentage = (df.isnull().mean() * 100)

# I defined a threshold of 30% for the maximum allowed missing values
threshold = 30

# To get the list of columns with missing values exceeding the threshold
columns_to_drop = missing_percentage[missing_percentage > threshold].index.tolist()

# dropping columns with excessive missing values from the DataFrame
df.drop(columns=columns_to_drop, inplace=True)
df

from numpy import NaN
for column in df.columns:
  df[column].replace(" ",np.NaN, inplace=True)

# Converting the column to float
df['TotalCharges'] = pd.to_numeric(df['TotalCharges']).astype(float)

# Identifying the categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns

# Identifying the numeric columns
numeric_columns = df.select_dtypes(exclude=['object']).columns

# Creating separate DataFrames for categorical and numeric data
categorical_data = df[categorical_columns]
numeric_data = df[numeric_columns]

categorical_columns

numeric_columns

#imputing the numeric and categorical data before the correlation
#Filling missing values in numeric data with the median.
from sklearn.impute import SimpleImputer
imp=SimpleImputer(strategy='median')
a=imp.fit_transform(numeric_data)
numeric_data_impute = pd.DataFrame(a,columns=numeric_columns)

# Filling missing values in categorical_data using forward fill (ffill)
categorical_data_imputed = categorical_data.fillna(method='ffill')

# Making sure the result is a DataFrame with the same column names
categorical_data_imputed = pd.DataFrame(categorical_data_imputed, columns=categorical_data.columns)

#Displaying the categorical_data_imputed and numeric_data_impute
categorical_data_imputed.info()
numeric_data_impute.info()

#Encoding the categorical data
from sklearn.preprocessing import LabelEncoder

# Initializing the LabelEncoder
label_encoder = LabelEncoder()

# Looping through each column in the imputed categorical data
for col in categorical_data_imputed.columns:
    categorical_data_imputed[col] = label_encoder.fit_transform(categorical_data_imputed[col])

categorical_data_imputed

numeric_data_impute

import pandas as pd
# Combining the imputed numeric and label-encoded categorical data into a single DataFrame
df = pd.concat([numeric_data_impute, categorical_data_imputed], axis=1)

#df = pd.concat([pd.DataFrame(a, columns=numeric_data.columns), categorical_data_imputed], axis=1)
df.info()
df

"""EXTRACTING RELEVANT FEATURES"""

from sklearn.ensemble import RandomForestRegressor

#'X' is my feature matrix and 'y' is my target variable
X = df.drop(['Churn'], axis=1)
y = df['Churn']

# Initializing a Random Forest Regressor
rf = RandomForestRegressor(random_state=32)

# Fitting the model to my data
rf.fit(X, y)

# Getting feature importances
feature_importances = rf.feature_importances_

# Creating a DataFrame to associate features with their importance scores
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

# Sorting the features by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Setting a threshold of 0.2 for feature importance
threshold = 0.02  # You can adjust this value as needed

# Selecting and storing feature names with importances above the given threshold
selected_features = feature_importance_df[feature_importance_df['Importance'] > threshold]['Feature'].tolist()

# Printing out the selected features
print("Selected Features based on Importance:")
print(selected_features)

"""USING EDA TO FIND OUT WHICH CUSTOMER PROFILES RELATE TO CHURNING A LOT"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# EDA: Relationship between Churn and other features

# 1. Churn distribution
sns.countplot(x='Churn', data=df)
plt.title('Churn Distribution')
plt.show()

# 2. Exploring all the selected features vs. Churn
features_to_explore = ['MonthlyCharges', 'TotalCharges', 'Contract', 'tenure', 'PaymentMethod',
                       'OnlineSecurity', 'gender', 'PaperlessBilling', 'TechSupport', 'Partner']

for feature in features_to_explore:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=feature, hue='Churn', data=df)
    plt.title(f'Churn Distribution by {feature}')
    plt.show()

# Correlation Matrix (to identify relationships between features)
correlation_matrix = df[features_to_explore + ['Churn']].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Dropping non-selected features from the dataset
non_selected_features = [col for col in df.columns if col not in selected_features]

df.drop(non_selected_features, axis=1, inplace=True)
df

#assigning df to x
x=df

#scaling x
from sklearn.preprocessing import StandardScaler

# Creating an instance of StandardScaler
scaler = StandardScaler()

# Fitting the scaler to my x and transforming them
scaled = scaler.fit_transform(x)

scaled

#pickling the model #creating the scalar model
import pickle
# Save the scaler
with open("scaled.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Splitting the scaled data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(scaled, y, test_size=0.2, random_state=42)

# Getting the feature names after scaling(this is to know the order when deploying)
feature_names_after_scaling = scaler.get_feature_names_out()

# Print the order of features
print("Feature Order after Scaling:", feature_names_after_scaling)

#Checking the shape (dimensions) of the variable x_train
x_train.shape

#Checking the shape(dimensions) of the variable x_test
x_test.shape

"""TRAINING USING FUNCTIONAL API AND TESTING THE MODEL'S ACCURACY"""

# Defining and training the Multi-Layer Perceptron model using the Functional API
import keras
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Keras Functional API model
input_layer = Input(shape=(x_train.shape[1],))
hidden_layer_1 = Dense(32, activation='relu')(input_layer)
hidden_layer_2 = Dense(24, activation='relu')(hidden_layer_1)
hidden_layer_3 = Dense(12, activation='relu')(hidden_layer_2)
output_layer = Dense(1, activation='sigmoid')(hidden_layer_3)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_test, y_test))

# Obtaining predicted probabilities for test set
y_prob = model.predict(x_test)

# Evaluate the model on train and test sets
_, train_accuracy = model.evaluate(x_train, y_train)
_, test_accuracy = model.evaluate(x_test, y_test)
print(f'Train Accuracy: {train_accuracy*100:.4f}')
print(f'Test Accuracy: {test_accuracy*100:.4f}')


# Plotting ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

"""EVALUATING THE MODEL'S AUC SCORE"""

# Making predictions on the test set
y_prob = model.predict(x_test)
auc_score = roc_auc_score(y_test, y_prob)
print("AUC Score:", auc_score)

X_Corr=scaled.copy()

!pip install tensorflow scikeras scikit-learn

!pip install scikeras

import numpy as np
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.layers import Dropout
from tensorflow.keras.constraints import MaxNorm
from sklearn.metrics import matthews_corrcoef
from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

"""BALANCING THE DATA"""

# Initializing the RandomOverSampler
oversampler = RandomOverSampler(sampling_strategy='auto', random_state=42)

# Applying random oversampling to the training data
X_train_resampled, y_train_resampled = oversampler.fit_resample(x_train, y_train)

# Printing the original and resampled class distribution
print("Original class distribution:", np.bincount(y_train))
print("Resampled class distribution:", np.bincount(y_train_resampled))

y_train.value_counts()

from sklearn.metrics import accuracy_score
from sklearn import metrics

num_classes=1
epochs=30
batch_size=10

"""TRAINING USING FUNCTIONAL API"""

#Creating the model function
def create_model(dropout_rate, weight_constraint,neurons,activation):
  # create modeloptimizer=optimizer
  input_layer = Input(shape=(x_train.shape[1],))
  hidden_layer_1 = Dense(32, activation='relu')(input_layer)
  hidden_layer_2 = Dense(24, activation='relu')(hidden_layer_1)
  hidden_layer_3 = Dense(12, activation='relu')(hidden_layer_2)
  output_layer = Dense(1, activation='sigmoid')(hidden_layer_3)

  m=Model(inputs=input_layer, outputs=output_layer)

  m.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
  return m

#Creating the model
model = KerasClassifier(model=create_model, epochs=epochs, batch_size=batch_size, verbose=0)
dropout_rate = [0.3, 0.5]
weight_constraint = [3.0, 5.0]
neurons = [20]
optimizer = ['SGD', 'Adam', 'RMSProp']
activation = ['relu']
param_grid = dict(model__dropout_rate=dropout_rate, model__weight_constraint=weight_constraint,
                  model__neurons=neurons,model__activation=activation)

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=inner_cv, scoring='accuracy')


# Initializing lists to store outer fold results
outer_scores = []
best_models = []

for train_idx, val_idx in outer_cv.split(X_train_resampled, y_train_resampled):
    X_train_outer, X_val_outer = X_train_resampled[train_idx], X_train_resampled[val_idx]
    y_train_outer, y_val_outer = y_train_resampled[train_idx], y_train_resampled[val_idx]

    # Perform hyperparameter tuning in the inner loop
    grid_search.fit(X_train_outer, y_train_outer)

    # Access the best estimator after fitting
    best_model = grid_search.best_estimator_

    best_models.append(best_model)

    # Evaluate the best model on the outer validation set
    y_pred_outer = best_model.predict(X_val_outer)
    accuracy = accuracy_score(y_val_outer, y_pred_outer)
    outer_scores.append(accuracy)

# Access the overall best estimator after the full search
final_best_model_mlp = grid_search.best_estimator_

"""CLASSIFICATION REPORT"""

from sklearn.metrics import classification_report

#Classification report
print("Outer CV Scores:", outer_scores)
print("Mean Accuracy:", np.mean(outer_scores))
print("Standard Deviation:", np.std(outer_scores))

# Train the final model on the entire training set with the best hyperparameters
final_best_model_mlp = grid_search.best_estimator_
print("The best estimator:",grid_search.best_estimator_, "\n")
final_best_model_mlp.fit(X_train_resampled, y_train_resampled,epochs=epochs, batch_size=batch_size, verbose=0)

"""CALCULATING THE AUC SCORE"""

# Evaluating the model and obtaining predicted probabilities
y_pred = final_best_model_mlp.predict(x_test)
fpr_mlp, tpr_mlp, _ = metrics.roc_curve(y_test, y_pred)
auc_mlp = round(metrics.roc_auc_score(y_test, y_pred), 4)
print("AUC:",auc_mlp)
y_pred=np.round(final_best_model_mlp.predict(x_test)).ravel()
print("\nCR by library method=\n",
          classification_report(y_test, y_pred))

"""CALCULATING MODEL'S ACCURACY"""

from sklearn.metrics import accuracy_score

# Evaluating the model and obtaining predicted probabilities
y_pred = final_best_model_mlp.predict(x_test)

# Converting predicted probabilities to binary predictions (0 or 1)
y_pred_binary = np.round(y_pred).ravel()

# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred_binary)
print("Accuracy:", accuracy)

"""OPTIMIZING FURTHER TO ACHIEVE BETTER RESULTS"""

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
from scikeras.wrappers import KerasClassifier
import tensorflow as tf
import numpy as np

# Defining my model creation function again for optimization
def create_model(dropout_rate, weight_constraint,neurons,activation):
  # create modeloptimizer=optimizer
  input_shape = (X_Corr.shape[1],)
  inputs = tf.keras.Input(shape=input_shape)
  input = tf.keras.layers.Dense((28)+neurons, activation=activation)(inputs)
  x= tf.keras.layers.Dropout(dropout_rate)(input)
  second=tf.keras.layers.Dense((12)+neurons, activation=activation)(x)
  x= tf.keras.layers.Dropout(dropout_rate)(second)
  third=tf.keras.layers.Dense((4)+neurons, activation=activation)(x)
  x= tf.keras.layers.Dropout(dropout_rate)(third)
  fourth=tf.keras.layers.Dense((-4)+neurons, activation=activation)(x)
  x= tf.keras.layers.Dropout(dropout_rate)(fourth)
  fifth=tf.keras.layers.Dense((-12)+neurons, activation=activation)(x)

# Adding output layer with softmax activation
  outputs = tf.keras.layers.Dense(num_classes, activation='sigmoid')(fifth)

# Creating the model
  m = tf.keras.Model(inputs=inputs, outputs=outputs)
  m.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])
  return m

# Setting seed for reproducibility
seed = 7
tf.random.set_seed(seed)

# Defining model and hyperparameter search space
model = KerasClassifier(model=create_model, epochs=epochs, batch_size=batch_size, verbose=0)
dropout_rate = [0.3, 0.5]
weight_constraint = [3.0, 5.0]
neurons = [20]
optimizer = ['SGD', 'Adam', 'RMSProp']
activation = ['relu']
param_grid = dict(model__dropout_rate=dropout_rate, model__weight_constraint=weight_constraint,
                  model__neurons=neurons, model__activation=activation)

# Defining cross-validation
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Initializing lists to store outer scores and the best models
outer_scores = []
best_models = []

# Outer cross-validation loop
for train_idx, val_idx in outer_cv.split(X_train_resampled, y_train_resampled):
    X_train_outer, X_val_outer = X_train_resampled[train_idx], X_train_resampled[val_idx]
    y_train_outer, y_val_outer = y_train_resampled[train_idx], y_train_resampled[val_idx]

    # Inner cross-validation loop for hyperparameter tuning
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=inner_cv, scoring='accuracy')
    grid_search.fit(X_train_outer, y_train_outer)

    # Accessing the best estimator after fitting
    best_model = grid_search.best_estimator_
    best_models.append(best_model)

    # Evaluating the best model on the outer validation set
    y_pred_outer = best_model.predict(X_val_outer)
    accuracy = accuracy_score(y_val_outer, y_pred_outer)
    outer_scores.append(accuracy)

# Accessing the overall best estimator after the full search
final_best_model_f = best_models[np.argmax(outer_scores)]

grid_search.best_estimator_

# Evaluating the model and obtaining predicted probabilities
y_pred = final_best_model_f.predict(x_test)
fpr_mlp, tpr_mlp, _ = metrics.roc_curve(y_test, y_pred)
auc_mlp = round(metrics.roc_auc_score(y_test, y_pred), 4)
print("AUC:",auc_mlp)
y_pred=np.round(final_best_model_mlp.predict(x_test)).ravel()
print("\nCR by library method=\n",
          classification_report(y_test, y_pred))

"""SAVING THE BEST MODEL"""

from keras.models import save_model

model = create_model(dropout_rate=0.3, weight_constraint=5.0, neurons=20, activation="relu")

# Manually setting the input names(This was to help in deployment. To get the right order)
model._name = 'my_model'
model._input_names = ['tenure', 'MonthlyCharges', 'TotalCharges', 'gender', 'Partner', 'OnlineSecurity', 'TechSupport', 'Contract', 'PaperlessBilling', 'PaymentMethod']

# Save the model to a file
save_model(model, "best_model.h5")
