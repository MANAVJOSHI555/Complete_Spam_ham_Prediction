from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Model aur vectorizer load karo (ye files aapke paas honi chahiye)
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']

    if message.strip() == "":
        return render_template("index.html", prediction="❌ Please enter a message!")

    # Message ko vectorize karo
    data = vectorizer.transform([message])

    # Prediction karo
    prediction = model.predict(data)[0]

    # Prediction ko readable banao
    label = "⚠️ Spam" if prediction == 1 else "✅ Real Mail(Ham)"

    # Result template ko bhejo
    return render_template("index.html", prediction=label)

if __name__ == "__main__":
    app.run(debug=True)

