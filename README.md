# Phishing-email-and-message-Detection-using-AI

# SecureScan - Fraud Detection System 🔍🛡️

A machine learning-powered web application that detects fraudulent messages and emails to protect users from social engineering attacks.

## Features ✨

-  Real-time fraud detection for messages and emails
-  Confidence percentage for each analysis
-  Detailed breakdown of detected scam patterns
-  Fully responsive design (works on all devices)
-  Privacy-focused (no data stored)
-  Supports multiple scam types:
  - Phishing attempts
  - Lottery scams
  - Fake job offers
  - KYC frauds
  - And many more...

## Technologies Used 💻

**Frontend:**
- HTML5, CSS3, JavaScript
- Font Awesome Icons
- Google Fonts (Poppins)

**Backend:**
- Python 3
- Flask (Web Framework)
- scikit-learn (Machine Learning)
- joblib (Model Persistence)

**Machine Learning:**
- TF-IDF Vectorization
- Naive Bayes Classifier
- Custom-trained model

## Installation & Setup ⚙️

### Prerequisites
- Python 3.8+
- pip package manager

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fraud-detection-app.git
   cd fraud-detection-app
2. Install dependencies:
pip install -r requirements.txt

4. Train the model (optional):
python train_model.py

6. Run the application:
python app.py

8. Open your browser and visit:
http://localhost:5000

## File Structure 📂
fraud-detection-app/
├── app.py                  # Flask application
├── train_model.py          # Model training script
├── fraud_dataset.csv       # Sample dataset
├── fraud_detection_model.pkl  # Pre-trained model
├── requirements.txt        # Dependencies
├── static/
│   ├── css/
│   │   └── style.css       # Custom styles
│   └── images/             # All static images
└── templates/
    └── index.html          # Main frontend template
Dataset Information 📊
The model is trained on a custom dataset.



