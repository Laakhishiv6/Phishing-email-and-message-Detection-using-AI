from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load("fraud_detection_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get user input
    user_input = request.form["text"]
    
    # Make prediction
    prediction = model.predict([user_input])[0]
    
    # Generate confidence score (simulated)
    confidence = np.random.uniform(85, 98) if prediction == "fraudulent" else np.random.uniform(75, 90)
    
    # Return result
    return jsonify({
        "result": prediction,
        "confidence": round(confidence, 1)
    })

if __name__ == "__main__":
    app.run(debug=True)