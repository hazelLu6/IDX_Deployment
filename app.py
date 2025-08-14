import streamlit as st
import pandas as pd
import pickle
from geopy.geocoders import Nominatim

# Load trained pipeline
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load your training feature names (important for building input DataFrame)
# If you saved X_train's columns during training, load them here:
with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# Geocoder setup
geolocator = Nominatim(user_agent="house_price_app")

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
        location = geolocator.geocode(address)
        if not location:
            st.error("Could not geocode the address. Please check and try again.")
        else:
            lat = location.latitude
            lon = location.longitude

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

            # Fill in any missing columns with defaults (needed for model pipeline)
            df_input = pd.DataFrame(input_dict)
            for col in feature_columns:
                if col not in df_input.columns:
                    df_input[col] = 0  # or sensible default

            df_input = df_input[feature_columns]  # match training order

            prediction = model.predict(df_input)[0]
            st.success(f"Estimated Price: ${prediction:,.0f}")

    except Exception as e:
        st.error(f"Error: {e}")
