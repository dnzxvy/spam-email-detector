from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib

model = joblib.load("spam_model.pkl")
vectorizer =joblib.load("tfidf_vectorizer.pkl")

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    email = data["email"]

    email_tfidf = vectorizer.transform([email])
    prediction = model.predict(email_tfidf)[0]
    if prediction == 1:
        result = "Spam"
    else:
        result = "Ham"

    return jsonify({
        "prediction": result
    })


if __name__ == "__main__":
    app.run(debug=True)