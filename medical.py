import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('E:\\kidney_disease.csv')

# Drop the 'id' column as it is not useful for prediction
data = data.drop(columns=['id'])

# Handle missing values
# For numerical columns, use the mean for imputation
num_imputer = SimpleImputer(strategy='mean')
num_cols = data.select_dtypes(include=['float64', 'int64']).columns
data[num_cols] = num_imputer.fit_transform(data[num_cols])

# For categorical columns, use the most frequent value for imputation
cat_imputer = SimpleImputer(strategy='most_frequent')
cat_cols = data.select_dtypes(include=['object']).columns
data[cat_cols] = cat_imputer.fit_transform(data[cat_cols])

# Convert categorical variables to numeric using Label Encoding
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Display the first few rows of the preprocessed dataset
data.head()
X = data.drop(columns=['classification'])
y = data['classification']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
print('Confusion Matrix:\n', conf_matrix)
data.isnull().sum()
import numpy as np

# Function to preprocess and predict based on user input
def predict_ckd(input_data):
    """
    Predict the likelihood of CKD based on input data.
    :param input_data: A dictionary containing the input data for each feature.
    :return: The prediction result ('ckd' or 'notckd').
    """
    
    # Ensure the input data has all required features
    input_values = []
    for feature in X.columns:  # Use the columns from the training set
        value = input_data.get(feature)
        if value is None:
            raise ValueError(f"Missing value for feature '{feature}'")
        input_values.append(value)
    
    # Convert the input into a numpy array and reshape for prediction
    input_array = np.array(input_values).reshape(1, -1)
    
    # Apply the same preprocessing steps (scaling)
    input_scaled = scaler.transform(input_array)
    
    # Predict using the trained model
    prediction = model.predict(input_scaled)
    
    # Decode the prediction to get the original label
    result = 'ckd' if prediction[0] == 1 else 'notckd'
    
    return result
# Example usage: Directly providing input data
user_input = {
    'age': 48.0, 'bp': 80.0, 'sg': 1.020, 'al': 1.0, 'su': 0.0, 'rbc': 0,
    'pc': 0, 'pcc': 1, 'ba': 0, 'bgr': 121.0, 
    'bu': 36.0, 'sc': 1.2, 'sod': 135.0, 'pot': 4.5, 'hemo': 15.4, 'pcv': 44,
    'wc': 7800, 'rc': 5.2, 'htn':1, 'dm':0, 'cad': 0, 'appet':1, 
    'pe': 0, 'ane': 0
}

# Make the prediction
result = predict_ckd(user_input)
print(f'The predicted classification is: {result}')
