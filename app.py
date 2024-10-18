from flask import Flask, request, render_template
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

models = {
    'Zoho': joblib.load('Zoho.pkl'),
    'Accenture': joblib.load('Accenture.pkl'),
    'IBM': joblib.load('IBM.pkl'),
    'Cognizant': joblib.load('Cognizant.pkl'),
}

mlbs = {
    'Zoho': joblib.load('mlb_zoho.pkl'),
    'Accenture': joblib.load('mlb_accenture.pkl'),
    'IBM': joblib.load('mlb_ibm.pkl'),
    'Cognizant': joblib.load('mlb_cognizant.pkl'),
}


def preprocess_input(data, company):
    Gender = data['Gender']
    CGPA = float(data['CGPA'])
    department = data['department']
    internship_experience = int(data['internship_experience'])
    Aptitude_score = int(data['Aptitude_score'])
    Communication_Skill_Score_out_of_10 = float(data['Communication_Skill_Score(out_of_10)'])
    skills = data['skills'].split(', ')

    input_df = pd.DataFrame({
        'Gender': [Gender],
        'CGPA': [CGPA],
        'department': [department],
        'internship_experience': [internship_experience],
        'Aptitude_score': [Aptitude_score],
        'Communication_Skill_Score(out_of_10)': [Communication_Skill_Score_out_of_10],
        'skills': [skills]
    })

    
    mlb = mlbs[company]

    
    skills_encoded = pd.DataFrame(mlb.transform(input_df['skills']), columns=mlb.classes_, index=input_df.index)
    input_df = input_df.join(skills_encoded).drop('skills', axis=1)

    
    label_encoder = LabelEncoder()
    input_df['Gender'] = label_encoder.fit_transform(input_df['Gender'])
    input_df['department'] = label_encoder.fit_transform(input_df['department'])


    return input_df


def predict_hiring(data, company):
    model = models.get(company)
    if not model:
        raise ValueError(f"Model for company '{company}' not found.")
    
    
    prediction = model.predict(data)
    if prediction == 1:
        return "There are chances you are likely to be hired !!!"
    else:
        return "The chances of hiring are not likely."


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user inputs from form
        user_inputs = request.form.to_dict()
        # Extract selected company from the form
        selected_company = user_inputs.pop('company')
        
        # Preprocess user inputs
        user_data = preprocess_input(user_inputs, selected_company)
        
        # Make prediction for the selected company
        prediction = predict_hiring(user_data, selected_company)
        
        # Pass prediction to the frontend
        return render_template('prediction.html', company=selected_company, prediction=prediction)
    except Exception as e:
        return f"An error occurred: {e}", 500

if __name__ == '__main__':
    app.run(debug=True)