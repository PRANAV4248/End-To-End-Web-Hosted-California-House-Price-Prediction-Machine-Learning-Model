from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("XGBoost_model.pkl")

# City dropdown mapping
CITY_COORDS = {
    "Los Angeles": (34.05, -118.25),
    "San Francisco": (37.77, -122.42),
    "San Diego": (32.72, -117.16),
    "San Jose": (37.34, -121.89)
}

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        # Get form data
        income = float(request.form["income"])        # yearly USD
        rooms = int(request.form["rooms"])
        bedrooms = int(request.form["bedrooms"])
        family = int(request.form["family"])
        age = int(request.form["age"])
        population = int(request.form["population"])
        city = request.form["city"]

        lat, lon = CITY_COORDS[city]

        # Prepare model input
        input_df = pd.DataFrame([{
            "MedInc": income / 10000,   # ✅ correct
            "HouseAge": age,
            "AveRooms": rooms,
            "AveBedrms": bedrooms,
            "Population": population,
            "AveOccup": family,
            "Latitude": lat,
            "Longitude": lon
        }])

        price = model.predict(input_df)[0] * 100000
        prediction = f"${price:,.2f}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
