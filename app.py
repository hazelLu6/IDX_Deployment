import streamlit as st
import pandas as pd
import pickle
import requests


# Load trained pipeline
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load your training feature names (important for building input DataFrame)
with open("features.pkl", "rb") as f:
    feature_columns = pickle.load(f)
    
print("Feature names loaded from pickle file:")
for name in feature_columns:
    print(name)

# Geocoder setup
def geocode_address(address):
    """
    Use PositionStack API to convert an address into (lat, lon).
    Requires POSITIONSTACK_API_KEY in Streamlit secrets.
    """
    api_key = st.secrets["POSITIONSTACK_API_KEY"]
    base_url = "http://api.positionstack.com/v1/forward"
    params = {
        "access_key": api_key,
        "query": address,
        "limit": 1
    }

    try:
        response = requests.get(base_url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()

        if "data" in data and len(data["data"]) > 0:
            location = data["data"][0]
            return location["latitude"], location["longitude"]
        else:
            st.error("No results found for the given address.")
            return None, None

    except requests.exceptions.RequestException as e:
        st.error(f"Error contacting PositionStack API: {e}")
        return None, None

st.title("🏡 California House Price Predictor")
st.write("Enter the address and property details to predict price.")

# Address input
address = st.text_input("Property Address", "1600 Amphitheatre Parkway, Mountain View, CA")

# Other features required by your model
living_area = st.number_input("Living Area (sq ft)", min_value=200, max_value=10000, value=1500)
bathrooms = st.number_input("Bathrooms", min_value=0, max_value=10, value=2)
bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=3)
stories = st.number_input("Stories", min_value=1, max_value=5, value=1)
garage_spaces = st.number_input("Garage Spaces", min_value=0, max_value=5, value=2)
lot_size_sqft = st.number_input("Lot Size (sq ft)", min_value=500, max_value=100000, value=5000)

# Facilities as yes/no
view_yn = st.checkbox("View")
pool_yn = st.checkbox("Private Pool")
attached_garage_yn = st.checkbox("Attached Garage")
fireplace_yn = st.checkbox("Fireplace")
new_construction_yn = st.checkbox("New Construction")

if st.button("Predict Price"):
    try:
        st.write("Secrets keys:", list(st.secrets.keys()))
        lat, lon = geocode_address(address)
        if lat is None or lon is None:
            st.stop()

        st.write(f"Latitude: {lat}, Longitude: {lon}")

        # Build input DataFrame with all model-required columns
        input_dict = {
            "Latitude": [lat],
            "Longitude": [lon],
            "LivingArea": [living_area],
            "BathroomsTotalInteger": [bathrooms],
            "BedroomsTotal": [bedrooms],
            "Stories": [stories],
            "GarageSpaces": [garage_spaces],
            "LotSizeSquareFeet": [lot_size_sqft],
            "ViewYN": [int(view_yn)],
            "PoolPrivateYN": [int(pool_yn)],
            "AttachedGarageYN": [int(attached_garage_yn)],
            "FireplaceYN": [int(fireplace_yn)],
            "NewConstructionYN": [int(new_construction_yn)]
        }

        # Fill in any missing columns with defaults
        df_input = pd.DataFrame(input_dict)
        for col in feature_columns:
            if col not in df_input.columns:
                df_input[col] = 0  # default value for missing columns

        # Reorder columns to match training data
        df_input = df_input[feature_columns]

        # Make prediction
        prediction = model.predict(df_input)[0]
        st.success(f"Estimated Price: ${prediction:,.0f}")

    except Exception as e:
        st.error(f"Error: {e}")