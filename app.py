import numpy as np  # For numerical operations
from flask import Flask, request, jsonify, render_template  # Flask web framework
import pickle  # For loading the saved machine learning model
import sklearn  # Ensure scikit-learn is installed for model compatibility
import numpy as np  # For numerical operations
from flask import Flask, request, jsonify, render_template  # Flask web framework
import pickle  # For loading the saved machine learning model
import sklearn  # Ensure scikit-learn is installed for model compatibility

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text = "The Predicted Crop is {}".format(prediction))

if __name__ == "__main__":
    flask_app.run(debug=True)