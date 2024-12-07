import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, classification_report
from imblearn.over_sampling import SMOTE

from google.colab import files
store = files.upload()

import io
import pandas as pd
df = pd.read_excel(io.BytesIO(store['Copper_Set.xlsx']))  
print(df)

REGRESSION_TARGET = 'selling_price'
CLASSIFICATION_TARGET = 'status'

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data
  
def preprocess_data(data):
    # Handle missing values
    data.replace('00000', np.nan, inplace=True)
    data.fillna(data.median(numeric_only=True), inplace=True)
    
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != CLASSIFICATION_TARGET:
            data[col] = LabelEncoder().fit_transform(data[col].astype(str))
  
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if abs(data[col].skew()) > 1:
            data[col] = np.log1p(data[col])
    return data

def split_data(data, target):
    X = data.drop(target, axis=1)
    y = data[target]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_regression_model(X_train, y_train):
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

def train_classification_model(X_train, y_train):
    model = ExtraTreesClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def streamlit_app():
    st.title("Copper Industry Prediction App")
    
    task = st.selectbox("Select Task", ["Regression", "Classification"])
    if task == "Regression":
        st.subheader("Enter Details for Selling Price Prediction")
    else:
        st.subheader("Enter Details for Status Prediction")
import streamlit as st

inputs = {}
for col in df.columns:
    if col not in [REGRESSION_TARGET, CLASSIFICATION_TARGET]:
        inputs[col] = st.text_input(f"Enter {col}")

  import streamlit as st

def streamlit_app():
    st.title("Copper Industry Prediction App")
    
    task = st.selectbox("Select Task", ["Regression", "Classification"])
    if task == "Regression":
        st.subheader("Enter Details for Selling Price Prediction")
    else:
        st.subheader("Enter Details for Status Prediction")
    
    inputs = {}
        if col not in [REGRESSION_TARGET, CLASSIFICATION_TARGET]:
            inputs[col] = st.text_input(f"Enter {col}")
    
    if st.button("Predict"):
        input_df = pd.DataFrame([inputs])
        input_df = preprocess_data(input_df)
        input_df = scaler.transform(input_df) 
        
        if task == "Regression":
            prediction = regression_model.predict(input_df)
            st.write(f"Predicted Selling Price: {prediction[0]:.2f}")
        else:
            prediction = classification_model.predict(input_df)
            st.write(f"Predicted Status: {prediction[0]}")

from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder 

def preprocess_data(data):
    data.replace('00000', np.nan, inplace=True)
    data.fillna(data.median(numeric_only=True), inplace=True)
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != CLASSIFICATION_TARGET:
            data[col] = LabelEncoder().fit_transform(data[col].astype(str)) 
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if abs(data[col].skew()) > 1:
            data[col] = np.log1p(data[col])
    
    return data

def preprocess_data(data):
    # Handle missing values
    data.replace('00000', np.nan, inplace=True)
    data.fillna(data.median(numeric_only=True), inplace=True)
  
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != CLASSIFICATION_TARGET:
            data[col] = LabelEncoder().fit_transform(data[col].astype(str)) 
  
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if abs(data[col].skew()) > 1:
            data[col] = np.log1p(data[col])
  
    for col in data.columns:
        try:
            data[col] = pd.to_numeric(data[col], errors='coerce')  
        except ValueError:
            print(f"Warning: Column '{col}' could not be converted to numeric. Check its contents.")  
  
    data.fillna(data.median(numeric_only=True), inplace=True) 
    
    return data

def streamlit_app():
    df = pd.read_excel('Copper_Set.xlsx') 

    REGRESSION_TARGET = 'selling_price'
    CLASSIFICATION_TARGET = 'status'

    data = preprocess_data(df.copy())  

    data = data[data[CLASSIFICATION_TARGET].isin(["WON", "LOST"])]
    data[CLASSIFICATION_TARGET] = data[CLASSIFICATION_TARGET].map({"WON": 1, "LOST": 0})

    scaler = StandardScaler() 
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = split_data(data, CLASSIFICATION_TARGET)
    X_train_clf = scaler.fit_transform(X_train_clf)
    X_test_clf = scaler.transform(X_test_clf)
    classification_model = train_classification_model(X_train_clf, y_train_clf)
    pickle.dump(classification_model, open("classification_model.pkl", "wb"))

def streamlit_app():
    df = pd.read_excel('Copper_Set.xlsx')  
    REGRESSION_TARGET = 'selling_price'
    CLASSIFICATION_TARGET = 'status'

    data = preprocess_data(df.copy())  
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = split_data(data, REGRESSION_TARGET) 
    regression_model = train_regression_model(X_train_reg, y_train_reg)

    reg_preds = regression_model.predict(X_test_reg)
    print(f"Regression RMSE: {np.sqrt(mean_squared_error(y_test_reg, reg_preds))}")

    streamlit_app()
    
