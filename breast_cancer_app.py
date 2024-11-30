# Step 2: Dataset Acquisition and Preparation
from sklearn.datasets import load_breast_cancer
import pandas as pd

# Load the dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target  # Add the target variable to the dataframe
df.to_csv("breast_cancer.csv", index=False)  # Save the dataset to a CSV file for reference

# Step 3: Data Preparation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data from the saved CSV
df = pd.read_csv("breast_cancer.csv")

# Separate features and target variable
X = df.drop(columns=["target"])
y = df["target"]

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling option (True to scale, False to use raw features)
USE_SCALING = True

if USE_SCALING:
    # Apply standard scaling to features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
else:
    scaler = None  # No scaling applied

# Step 4: Feature Selection
from sklearn.feature_selection import SelectKBest, f_classif

# Select top 10 features using ANOVA F-statistic
selector = SelectKBest(score_func=f_classif, k=10)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Step 5: Grid Search Cross-Validation for Model Tuning
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

# Define the model
model = MLPClassifier(max_iter=1000, random_state=42)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (100, 50)],  # Different layer configurations
    'activation': ['relu', 'tanh'],  # Activation functions
    'solver': ['adam', 'sgd'],  # Optimization algorithms
    'alpha': [0.0001, 0.001]  # Regularization parameter
}

# Perform grid search with 5-fold cross-validation
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_selected, y_train)  # Train the model with selected features
print(f"Best Parameters: {grid_search.best_params_}")

# Step 6: Implementing and Evaluating the ANN Model
from sklearn.metrics import classification_report, accuracy_score

# Use the best model found during grid search
best_model = grid_search.best_estimator_
best_model.fit(X_train_selected, y_train)  # Train the best model on the training data

# Make predictions on the test set
y_pred = best_model.predict(X_test_selected)

# Print evaluation metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# Step 7: Save Model, Scaler, and Features
import joblib

# Save the trained model
joblib.dump(best_model, "best_model.pkl")

# Save the scaler (if scaling is used)
if USE_SCALING:
    joblib.dump(scaler, "scaler.pkl")

# Save selected feature names
joblib.dump(selector.get_feature_names_out(), "feature_names.pkl")

# Step 8: Building a Streamlit App
import streamlit as st

# Load the saved model, scaler, and feature names
model = joblib.load("best_model.pkl")

if USE_SCALING:
    scaler = joblib.load("scaler.pkl")  # Load the scaler if scaling was applied

feature_names = joblib.load("feature_names.pkl")  # Load the selected feature names

# Streamlit App Title
st.title("Breast Cancer Prediction App")

# Create input fields for each selected feature
inputs = {feature: st.number_input(feature, value=0.0) for feature in feature_names}

# Prediction Button
if st.button("Predict"):
    # Convert input to a DataFrame
    input_df = pd.DataFrame([inputs])
    
    # Apply scaling if required
    if USE_SCALING:
        input_scaled = scaler.transform(input_df)
    else:
        input_scaled = input_df.values
    
    # Make prediction using the loaded model
    prediction = model.predict(input_scaled)
    
    # Display the result
    st.write("Prediction:", "Malignant" if prediction[0] else "Benign")
    
