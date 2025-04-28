import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Step 1: Create the dataset (if it doesn't exist)
def create_dataset():
    data = {
        "text": [
            # Fraudulent examples
            "Congratulations! You've won a $1000 Walmart gift card. Click here to claim your prize.",
            "Your account has been suspended. Please update your KYC details to avoid permanent suspension.",
            "Earn ₹50,000 per month from home. Click here to apply now!",
            "Your electricity bill is overdue. Pay now to avoid disconnection.",
            "You have received a payment of $500. Click here to claim.",
            "Your SIM card will be blocked in 24 hours. Update your details now.",
            "Hi, I'm calling from Microsoft support. Your computer has a virus.",
            "You have been selected for an internship at Google. Click here to accept.",
            "Claim your lottery prize of $1,000,000! Click here to proceed.",
            "Your bank account has been compromised. Click here to secure it.",

            # Non-fraudulent examples
            "Hi, can you send me the report by EOD?",
            "Your package has been delivered. Track your order here.",
            "Reminder: Your meeting is scheduled for 3 PM today.",
            "Thanks for your payment. Your transaction ID is 123456.",
            "Your order #12345 has been shipped. Track it here.",
            "Hi, let's catch up for coffee this weekend.",
            "Your monthly statement is ready. Download it here.",
            "Your subscription is about to expire. Renew now to continue.",
            "Your appointment is confirmed for tomorrow at 10 AM.",
            "Your feedback is important to us. Please fill out this survey."
        ],
        "label": [
            "fraudulent", "fraudulent", "fraudulent", "fraudulent", "fraudulent",
            "fraudulent", "fraudulent", "fraudulent", "fraudulent", "fraudulent",
            "non-fraudulent", "non-fraudulent", "non-fraudulent", "non-fraudulent", "non-fraudulent",
            "non-fraudulent", "non-fraudulent", "non-fraudulent", "non-fraudulent", "non-fraudulent"
        ]
    }

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv("fraud_dataset.csv", index=False)
    print("Dataset created and saved as 'fraud_dataset.csv'.")

# Step 2: Train the model
def train_model():
    # Load dataset
    df = pd.read_csv("fraud_dataset.csv")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

    # Create a pipeline with TF-IDF and Naive Bayes
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())

    # Train the model
    model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)
    print("\nModel Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save the trained model
    joblib.dump(model, "fraud_detection_model.pkl")
    print("\nModel trained and saved as 'fraud_detection_model.pkl'.")

# Main function
if __name__ == "__main__":
    # Step 1: Create the dataset
    create_dataset()

    # Step 2: Train the model
    train_model()