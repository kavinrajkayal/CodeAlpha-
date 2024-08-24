# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv("C:\\Users\\siyam\\Downloads\\archive (14).zip")
data
# Preprocessing
X = data.drop('Credit Score', axis=1)
y = data['Credit Score']
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define preprocessing for numerical and categorical data
numeric_features = ['Income', 'Age']
categorical_features = ['Marital Status', 'Education']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])
# Define the model
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier(random_state=42))])

# Train the model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
# Evaluate the model
print(classification_report(y_test, y_pred))
# Function to predict new data
def predict_new_data(user_input):
    """
    Predict credit score based on user input.
    
    Parameters:
    user_input (dict): Dictionary with keys as feature names and values as input data.
    
    Returns:
    str: Prediction result.
    """
    # Convert user input to DataFrame
    input_df = pd.DataFrame([user_input])
    
    # Preprocess input data
    processed_input = model.named_steps['preprocessor'].transform(input_df)
    
    # Make prediction
    prediction = model.named_steps['classifier'].predict(processed_input)
    
    return prediction[0]

# Example usage
user_input = {
    'Income': 50000,
    'Age': 30,
    'Marital Status': 'Single',
    'Education': 'Bachelor'
}

prediction = predict_new_data(user_input)
print(f'The predicted credit score category is: {prediction}')