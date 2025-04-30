import pickle
from flask import Flask, request, jsonify
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the saved model
filename = 'best_job_title_prediction_model.pkl'
loaded_model = pickle.load(open(filename, 'rb'))

# Define known categories (same used in Streamlit app)
company_options = [
    'Accenture', 'Oracle', 'Siemens', 'BNY Mellon', 'CoinDCX', 'Rave Technologies',
    'HealthSpring', 'Citibank, N.A', 'Snaphunt', 'Duff & Phelps', 'Credit Suisse',
    'Prodair Air Products', 'Ubisoft', 'CompuCom', 'Kraftmaid Services India',
    'Method Studios', 'Company3 Method India Private Limited', 'Eversendai', 'Shell',
    'NatWest Group', 'Sona Comstar', 'RRD', 'Thinksynq Solutions', 'Icon Clinical Research',
    'Aspire Systems'
]

location_options = [
    'Mumbai ', 'Mumbai (All Areas)', 'Mumbai (All Areas), Hyderabad/Secunderabad, Pune, Chennai, Delhi / NCR, Bangalore/Bengaluru',
    'Hyderabad/Secunderabad, Pune, Chennai, Delhi / NCR, Bangalore/Bengaluru', 'Hyderabad/Secunderabad', 'Pune', 'Chennai', 'Delhi / NCR',
    'Bangalore/Bengaluru', 'Chennai(Teynampet)', 'Chennai(Kodambakkam)', 'Mumbai, Gurgaon/Gurugram, Aurangabad, Vadodara',
    'Pune, Hyderabad/Secunderabad, Chennai, Delhi / NCR, Bangalore/Bengaluru, Mumbai (All Areas)',
    'Chennai, Hyderabad/Secunderabad, Pune, Delhi / NCR, Bangalore/Bengaluru, Mumbai (All Areas)',
    'Chennai(Ekkaduthangal)', 'Chennai(Kodambakkam), Kodambakkam'
]

salary_options = ['0-3 LPA', '3-6 LPA', '6-10 LPA', '10-15 LPA', '15+ LPA']  # Example

# Fit LabelEncoders
le_company = LabelEncoder()
le_company.fit(company_options)

le_location = LabelEncoder()
le_location.fit(location_options)

le_salary = LabelEncoder()
le_salary.fit(salary_options)

# You may have to adjust this if the model expects encoded targets
le_title = LabelEncoder()
le_title.classes_ = loaded_model.classes_  # Assuming model has .classes_ attribute

# Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_job_title():
    try:
        data = request.get_json()
        experience = float(data['experience'])
        reviews = int(data['reviews'])
        ratings = float(data['ratings'])
        company = data['company']
        location = data['location']
        salary = data['salary']

        # Encode inputs
        try:
            company_enc = le_company.transform([company])[0]
        except ValueError:
            company_enc = -1
        try:
            location_enc = le_location.transform([location])[0]
        except ValueError:
            location_enc = -1
        try:
            salary_enc = le_salary.transform([salary])[0]
        except ValueError:
            salary_enc = -1

        # Model input
        input_features = [[experience, reviews, ratings, company_enc, location_enc, salary_enc]]
        prediction = loaded_model.predict(input_features)

        predicted_job_title = le_title.inverse_transform(prediction)[0]

        return jsonify({'predicted_job_title': predicted_job_title})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
