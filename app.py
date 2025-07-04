import pickle
import pandas as pd
import streamlit as st

# Load trained pipeline
with open('housing_pipeline.pkl', 'rb') as f:
    model = pickle.load(f)

# Streamlit
st.title("🏠 House Price Predictor")

# User inputs
area = st.number_input("Area (sq-ft):", min_value=100, max_value=20000, value=7420)
bedrooms = st.selectbox("Number of Bedrooms:", [1, 2, 3, 4, 5, 6], index=2)
bathrooms = st.selectbox("Number of Bathrooms:", [1, 2, 3, 4], index=1)
stories = st.selectbox("Number of Stories:", [1, 2, 3, 4], index=1)
parking = st.selectbox("Number of Parkings:", [0, 1, 2, 3], index=1)
furnishingstatus = st.selectbox("Furnishing Status:", ['furnished', 'semi-furnished', 'unfurnished'], index=1)

st.subheader("Other Features")
mainroad = st.checkbox('Beside Main Road')
guestroom = st.checkbox('Guest Room')
basement = st.checkbox('Basement')
hotwaterheating = st.checkbox('Hot Water Heating')
airconditioning = st.checkbox('Air Conditioning')
prefarea = st.checkbox('Preferred Area')

# Predict
if st.button("Predict"):
    # Assemble input
    input_dict = {
        'area': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'stories': stories,
        'parking': parking,
        'furnishingstatus': furnishingstatus,
        'mainroad': 'yes' if mainroad else 'no',
        'guestroom': 'yes' if guestroom else 'no',
        'basement': 'yes' if basement else 'no',
        'hotwaterheating': 'yes' if hotwaterheating else 'no',
        'airconditioning': 'yes' if airconditioning else 'no',
        'prefarea': 'yes' if prefarea else 'no'
    }
    input_df = pd.DataFrame([input_dict])

    # Predict using pipeline (automatically handles preprocessing)
    price = model.predict(input_df)[0]

    st.success(f"Estimated Price: ₹{int(price):,}")