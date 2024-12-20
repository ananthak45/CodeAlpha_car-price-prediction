import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
from flask import Flask, request, render_template
import os

# Load the dataset
data_path = 'car_data.csv'
data = pd.read_csv(data_path)

# Preprocessing
data['Current_Year'] = 2024
data['Car_Age'] = data['Current_Year'] - data['Year']
data = data.drop(['Year', 'Current_Year'], axis=1)

# Encode categorical variables
data = pd.get_dummies(data, drop_first=True)

# Features and target
X = data.drop('Selling_Price', axis=1)
y = data['Selling_Price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Save the model
model_filename = 'random_forest_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

# Flask application
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collecting input from the form
        present_price = float(request.form['present_price'])
        kms_driven = float(request.form['kms_driven'])
        fuel_type = request.form['fuel_type'].capitalize()
        seller_type = request.form['seller_type'].capitalize()
        transmission = request.form['transmission'].capitalize()
        owner = int(request.form['owner'])

        # Create a DataFrame for the input data
        input_features = pd.DataFrame({
            'Present_Price': [present_price],
            'Kms_Driven': [kms_driven],
            'Owner': [owner],
            'Car_Age': [2024 - 2023]  # Replace 2023 with the actual year of the car
        })

        # Add dummy variables for categorical columns
        input_features['Fuel_Type_Diesel'] = 1 if fuel_type == 'Diesel' else 0
        input_features['Fuel_Type_Petrol'] = 1 if fuel_type == 'Petrol' else 0
        input_features['Seller_Type_Individual'] = 1 if seller_type == 'Individual' else 0
        input_features['Transmission_Manual'] = 1 if transmission == 'Manual' else 0

        # Ensure all columns match the training data
        for col in X.columns:
            if col not in input_features.columns:
                input_features[col] = 0  # Add missing columns with default value 0
        
        input_features = input_features[X.columns]  # Align the column order with training data

        # Load the model
        with open(model_filename, 'rb') as file:
            loaded_model = pickle.load(file)

        # Predict
        prediction = loaded_model.predict(input_features)

        # Return the prediction to the template
        return render_template(
            'index.html', 
            prediction_text=f'The predicted selling price of the car is: {prediction[0]:.2f} lakhs'
        )

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    if not os.path.exists("templates"):
        os.makedirs("templates")
    app.run(debug=True)
