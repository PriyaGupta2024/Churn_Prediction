from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)  # Fixed "__name__" to __name__ without quotes

# Load the model
try:
    with open(r'C:\AI & ML\Task3\Churning_model.pkl', 'rb') as f:  # Use raw string (r'') for file path on Windows
        model = pickle.load(f)
except Exception as e:
    raise FileNotFoundError(f"Error loading the model: {str(e)}")

# Home route
@app.route("/")
def home_page():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Retrieve form data
            data = request.form

            # Helper function to map categorical string values to numeric values
            def map_value(value, mapping):
                if value:
                    return mapping.get(value.lower(), None)
                return None  # Return None if input is missing or invalid

            # Define mappings for categorical variables
            mappings = {
                "YesNo": {"yes": 1, "no": 0},
                "PhoneService": {"no phone service": 0, "yes": 1, "no": 0},
                "InternetService": {"dsl": 1, "fiber optic": 2, "no": 0},
                "Contract": {"month-to-month": 0, "one year": 1, "two year": 2},
                "PaymentMethod": {
                    "electronic check": 0,
                    "mailed check": 1,
                    "bank transfer (automatic)": 2,
                    "credit card (automatic)": 3,
                }
            }

            # Convert input data using the mappings
            SeniorCitizen = map_value(data.get('SeniorCitizen'), mappings["YesNo"])
            Partner = map_value(data.get('Partner'), mappings["YesNo"])
            Dependents = map_value(data.get('Dependents'), mappings["YesNo"])
            Tenure = int(data.get('Tenure')) if data.get('Tenure') else 0
            PhoneService = map_value(data.get('PhoneService'), mappings["PhoneService"])
            MultipleLines = map_value(data.get('MultipleLines'), mappings["PhoneService"])
            InternetService = map_value(data.get('InternetService'), mappings["InternetService"])
            OnlineSecurity = map_value(data.get('OnlineSecurity'), mappings["YesNo"])
            OnlineBackup = map_value(data.get('OnlineBackup'), mappings["YesNo"])
            DeviceProtection = map_value(data.get('DeviceProtection'), mappings["YesNo"])
            TechSupport = map_value(data.get('TechSupport'), mappings["YesNo"])
            StreamingTV = map_value(data.get('StreamingTV'), mappings["YesNo"])
            StreamingMovies = map_value(data.get('StreamingMovies'), mappings["YesNo"])
            Contract = map_value(data.get('Contract'), mappings["Contract"])
            PaperlessBilling = map_value(data.get('PaperlessBilling'), mappings["YesNo"])
            PaymentMethod = map_value(data.get('PaymentMethod'), mappings["PaymentMethod"])
            MonthlyCharges = float(data.get('MonthlyCharges')) if data.get('MonthlyCharges') else 0.0
            TotalCharges = float(data.get('TotalCharges')) if data.get('TotalCharges') else 0.0

            # Check for missing or invalid inputs
            if None in [
                SeniorCitizen, Partner, Dependents, PhoneService, MultipleLines,
                InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
                TechSupport, StreamingTV, StreamingMovies, Contract,
                PaperlessBilling, PaymentMethod
            ]:
                return render_template('index.html', output_user="Error: Invalid or missing inputs. Please check the form.")

            # Prepare input for the model
            user_input = np.array([[SeniorCitizen, Partner, Dependents, Tenure, PhoneService, MultipleLines,
                                    InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
                                    TechSupport, StreamingTV, StreamingMovies, Contract,
                                    PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges]])

            # Perform prediction
            model_output = model.predict(user_input)
            output_user = 'The customer is likely to churn.' if model_output[0] == 1 else ' The customer is likely to stay'

            return render_template('index.html', output_user=output_user)

        except Exception as e:
            # Catch unexpected errors
            return render_template('index.html', output_user=f"An error occurred: {str(e)}")


# Run the app
if __name__ == "__main__":
    app.run(debug=True)
