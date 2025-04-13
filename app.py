from flask import Flask, render_template, request
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

app = Flask(__name__)

# Load model, tokenizer, and label encoder
model = load_model("sentiment_lstm_model.h5")
tokenizer = joblib.load("tokenizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    user_input = ""
    if request.method == "POST":
        user_input = request.form["tweet"]
        sequence = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(sequence, maxlen=100)
        pred = model.predict(padded)
        label_index = np.argmax(pred)
        prediction = label_encoder.inverse_transform([label_index])[0]

    return render_template("index.html", prediction=prediction, user_input=user_input)

if __name__ == "__main__":
    app.run(debug=True)