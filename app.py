import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import streamlit as st
import pandas as pd

from ML_LAB.pipeline.prediction import PredictionPipeline

st.set_page_config(page_title="Hotel Booking Prediction", layout="wide")
st.title("🏨 Hotel Booking Cancellation Predictor - LIVE ")

# Form for User Input
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        no_of_adults = st.number_input("No of Adults", min_value=0, value=1)
        no_of_children = st.number_input("No of Children", min_value=0, value=0)
        no_of_weekend_nights = st.number_input("Weekend Nights", min_value=0, value=0)
        no_of_week_nights = st.number_input("Week Nights", min_value=0, value=2)
        
    with col2:
        type_of_meal_plan = st.selectbox("Meal Plan", ['Meal Plan 1', 'Meal Plan 2', 'Not Selected']) 
        required_car_parking_space = st.selectbox("Car Parking Required?", [0, 1])
        room_type_reserved = st.selectbox("Room Type", ['Room_Type 1', 'Room_Type 2', 'Room_Type 4'])
        
        # 🔹 Replaced number_input with slider for better UI
        lead_time = st.slider("Lead Time (Days before arrival)", 0, 365, 10)

    with col3:
        market_segment_type = st.selectbox("Market Segment", ['Online', 'Offline', 'Corporate', 'Aviation'])
        avg_price_per_room = st.number_input("Avg Price per Room", min_value=0.0, value=100.0)
        no_of_special_requests = st.number_input("Special Requests", min_value=0, value=0)
        repeated_guest = st.selectbox("Is Repeated Guest?", [0, 1])

    # 🔹 Dynamic Total Nights display inside the form
    total_nights = no_of_weekend_nights + no_of_week_nights
    st.info(f"📌 Total Stay Nights: **{total_nights}**")

    # 🔹 Auto-calculation of average cost per person
    total_guests = no_of_adults + no_of_children
    avg_per_person = avg_price_per_room / max(total_guests, 1)
    st.info(f"💰 Avg Price per Person: **₹{avg_per_person:.2f}**")

    submit_btn = st.form_submit_button("Predict Status")

if submit_btn:
    # Data prepare kar rahe hain same format mein jaisa training me tha
    data = {
        'no_of_adults': [no_of_adults],
        'no_of_children': [no_of_children],
        'no_of_weekend_nights': [no_of_weekend_nights],
        'no_of_week_nights': [no_of_week_nights],
        'type_of_meal_plan': [type_of_meal_plan],
        'required_car_parking_space': [required_car_parking_space],
        'room_type_reserved': [room_type_reserved],
        'lead_time': [lead_time],
        'arrival_year': [2018],  # Dummy values jo feature importance me kam hain
        'arrival_month': [1],
        'arrival_date': [1],
        'market_segment_type': [market_segment_type],
        'repeated_guest': [repeated_guest],
        'no_of_previous_cancellations': [0],
        'no_of_previous_bookings_not_canceled': [0],
        'avg_price_per_room': [avg_price_per_room],
        'no_of_special_requests': [no_of_special_requests]
    }
    
    try:
        pipeline = PredictionPipeline()

        # 🔹 Add prediction spinner for interactivity
        with st.spinner("⏳ Predicting… Please wait"):
            result = pipeline.predict(data)
        
        if result[0] == 1:
            st.error("⚠️ Prediction: Booking will be CANCELED")
        else:
            st.success("✅ Prediction: Booking will NOT be Canceled (Confirmed)")
            
    except Exception as e:
        st.error(f"Error occurred: {e}")
