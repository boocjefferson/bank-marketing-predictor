from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# 1. Load the saved Pipeline, Encoder, and Columns
# The 'bank_model.pkl' now contains both SMOTE logic and Logistic Regression
model_pipeline = joblib.load('bank_model.pkl')
le = joblib.load('label_encoder.pkl')
model_columns = joblib.load('model_columns.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = ""
    prob_yes = None  # Initialize to None for the GET request
    
    if request.method == 'POST':
        # 2. Capture data from the HTML Form
        user_input = {
            'age': int(request.form['age']),
            'job': request.form['job'],
            'marital': request.form['marital'],
            'education': request.form['education'],
            'default': request.form['default'],
            'balance': int(request.form['balance']),
            'housing': request.form['housing'],
            'loan': request.form['loan'],
            'contact': request.form['contact'],
            'day': int(request.form['day']),
            'month': request.form['month'],
            'duration': int(request.form['duration']),
            'campaign': int(request.form['campaign']),
            'pdays': int(request.form['pdays']),
            'previous': int(request.form['previous']),
            'poutcome': request.form['poutcome']
        }
        
        # 3. Process the input into a DataFrame
        input_df = pd.DataFrame([user_input])
        input_dummies = pd.get_dummies(input_df)
        input_dummies = input_dummies.reindex(columns=model_columns, fill_value=0)
        input_dummies = input_dummies.astype(int)
        
        # --- X-RAY DEBUG MODE ---
        # We look inside the pipeline to get the probability scores
        probs = model_pipeline.predict_proba(input_dummies)[0]
        print("\n" + "="*40)
        print("🔍 X-RAY MODE: LOGISTIC REGRESSION MATH")
        print(f"Confidence: {probs[0]*100:.1f}% NO | {probs[1]*100:.1f}% YES")
        print("="*40 + "\n")
        
        # 4. Make the final prediction and extract percentage
        pred = model_pipeline.predict(input_dummies)
        result = le.inverse_transform(pred)[0] 
        
        # Grab the 'Yes' probability (index 1) and turn it into a clean percentage
        prob_yes = round(probs[1] * 100, 2)
        
        if result == 'yes':
            prediction_text = "Result: This client is LIKELY to subscribe!"
        else:
            prediction_text = "Result: This client is UNLIKELY to subscribe."

    # Pass both the text and the probability to the template
    return render_template('index.html', prediction_text=prediction_text, probability=prob_yes)

if __name__ == '__main__':
    app.run(debug=True)