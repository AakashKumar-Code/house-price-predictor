from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pickle

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    prediction_text = request.args.get('prediction_text')
    return render_template('index.html', prediction_text=prediction_text)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        feature_values = [float(x) for x in request.form.values()]
        input_features = np.array([feature_values])

        prediction = model.predict(input_features)
        predicted_price = round(prediction[0] * 100000)  # Round to nearest integer

        formatted_price = f"${predicted_price:,}"
        return redirect(url_for('home', prediction_text=f"Estimated House Price: {formatted_price}"))

    except Exception as e:
        error_message = f"An error occurred: {e}"
        return redirect(url_for('home', prediction_text=error_message))

if __name__ == '__main__':
    app.run(debug=True)
