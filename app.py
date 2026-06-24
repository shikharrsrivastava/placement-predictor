from flask import Flask, request

import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():

    return """
    <h1>Placement Predictor</h1>

    <form method="POST" action="/predict">

        CGPA:
        <input type="number" step="0.01" name="cgpa">

        <br><br>

        IQ:
        <input type="number" name="iq">

        <br><br>

        <button type="submit">
            Predict
        </button>

    </form>
    """

@app.route("/predict", methods=["POST"])
def predict():

    cgpa = float(request.form["cgpa"])
    iq = float(request.form["iq"])

    arr = np.array([[cgpa, iq]])

    arr = scaler.transform(arr)

    result = model.predict(arr)[0]

    if result == 1:
        return "<h1>Placed</h1>"

    return "<h1>Not Placed</h1>"

if __name__ == "__main__":
    app.run(debug=True)