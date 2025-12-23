"This python file is for testing of the california house prediction model in python terminal in any IDE like VS code."

import pandas as pd
import joblib
 
# ------------------------------- Load trained model -------------------------------

model = joblib.load("XGBoost_model.pkl")

# City → Latitude, Longitude
CITY_COORDS = {
    "los angeles": (34.05, -118.25),
    "san francisco": (37.77, -122.42),
    "san diego": (32.72, -117.16),
    "san jose": (37.34, -121.89)
}

# ------------------------------- Convert user input to model input -------------------------------

def prepare_features(
    yearly_income,     # ✅ yearly income in USD
    rooms,
    bedrooms,
    family_members,
    house_age,
    population,
    city
):
    lat, lon = CITY_COORDS[city.lower()]

    return pd.DataFrame([{
        "MedInc": yearly_income / 10000, 
        "HouseAge": house_age,
        "AveRooms": rooms,
        "AveBedrms": bedrooms,
        "Population": population,
        "AveOccup": family_members,
        "Latitude": lat,
        "Longitude": lon
    }])

# ------------------------------- User Input -------------------------------

print("\n🏠 California House Price Predictor\n")

yearly_income = float(input("Yearly income in USD ($): "))
rooms = int(input("Total rooms: "))
bedrooms = int(input("Bedrooms: "))
family_members = int(input("Family members: "))
house_age = int(input("House age (years): "))
population = int(input("Nearby population estimate: "))
city = input("City (Los Angeles / San Francisco / San Diego / San Jose): ")

# ------------------------------- Prediction -------------------------------

input_df = prepare_features(
    yearly_income,
    rooms,
    bedrooms,
    family_members,
    house_age,
    population,
    city
)

prediction = model.predict(input_df)[0]

print(f"\n💰 Estimated House Price: ${prediction * 100000:,.2f}")