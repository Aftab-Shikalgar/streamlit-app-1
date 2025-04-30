import pickle
from flask import Flask, request, jsonify

# Load the trained model
with open('best_job_title_prediction_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Load LabelEncoders
with open('le_company.pkl', 'rb') as f:
    le_company = pickle.load(f)

with open('le_location.pkl', 'rb') as f:
    le_location = pickle.load(f)

with open('le_salary.pkl', 'rb') as f:
    le_salary = pickle.load(f)

with open('le_title.pkl', 'rb') as f:
    le_title = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_job_title():
    try:
        data = request.get_json()

        # Extract features from request
        experience = float(data['experience'])
        reviews = int(data['reviews'])
        ratings = float(data['ratings'])
        company = data['company']
        location = data['location']
        salary = data['salary']

        # Encode categorical variables with fallback for unknown values
        try:
            company_enc = le_company.transform([company])[0]
        except ValueError:
            company_enc = -1  # or a neutral default

        try:
            location_enc = le_location.transform([location])[0]
        except ValueError:
            location_enc = -1

        try:
            salary_enc = le_salary.transform([salary])[0]
        except ValueError:
            salary_enc = -1

        # Form feature vector
        input_features = [[experience, reviews, ratings, company_enc, location_enc, salary_enc]]

        # Make prediction
        prediction = loaded_model.predict(input_features)
        predicted_title = le_title.inverse_transform(prediction)[0]

        return jsonify({'predicted_job_title': predicted_title})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)  # Important to avoid signal/thread issue
