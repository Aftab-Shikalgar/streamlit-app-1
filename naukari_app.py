# prompt: create api for inference

import pickle
from flask import Flask, request, jsonify

# Load the saved model
filename = 'best_job_title_prediction_model.pkl'
loaded_model = pickle.load(open(filename, 'rb'))

# Load LabelEncoders (replace with your actual loading method)
le_company = globals().get('le_company')
le_location = globals().get('le_location')
le_salary = globals().get('le_salary')
le_title = globals().get('le_title')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_job_title():
    try:
        data = request.get_json()
        experience = float(data['experience'])
        reviews = int(data['reviews'])
        ratings = float(data['ratings'])
        company = data['company']  # Assuming 'company' is a string
        location = data['location']
        salary = data['salary']  # Assuming salary is a string

        # Preprocessing steps
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

        input_features = [[experience, reviews, ratings, company_enc, location_enc, salary_enc]]
        prediction = loaded_model.predict(input_features)
        predicted_job_title = le_title.inverse_transform(prediction)[0]

        return jsonify({'predicted_job_title': predicted_job_title})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
