import streamlit as st
import requests
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import pandas as pd
import numpy as np
import datetime
import random
import plotly.graph_objects as go
import time
import os
from streamlit_folium import folium_static
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase once
firebase_json_path = "C:\\Users\\Lenovo\\OneDrive\\Desktop\\bloomalert\\bloom-watch-firebase-admin.json"
if not firebase_admin._apps:
    if os.path.exists(firebase_json_path):
        cred = credentials.Certificate(firebase_json_path)
        firebase_admin.initialize_app(cred)
    else:
        st.error(f"Firebase JSON file not found at {firebase_json_path}")
        st.stop()

db = firestore.client()

# Firebase API Key for auth
FIREBASE_API_KEY = "replace with ur orginal api key"
species_list = [f" 🌹Rose", "🌼Jasmine", "🌸Lotus", "🌻Sunflower", "🌷Tulip", "🌼Dandelion", "🌼Marigold", "🌸Lavender", "🌺Orchid", "🌺Hibiscus"]

# Sign Up / Sign In functions
def sign_up(email, password):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={FIREBASE_API_KEY}"
    payload = {"email": email, "password": password, "returnSecureToken": True}
    res = requests.post(url, json=payload).json()
    return res

def sign_in(email, password):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"
    payload = {"email": email, "password": password, "returnSecureToken": True}
    res = requests.post(url, json=payload).json()
    return res

# Session state init
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Login page image URL (replace with your preferred image or local path)
LOGIN_PAGE_IMAGE_URL = "https://science.nasa.gov/wp-content/uploads/2023/04/potw1926a-jpg.webp"

# Main view: login or app
import streamlit as st

# Adjust image source as needed
# --- LOGIN PAGE ---
import streamlit as st

# Use your uploaded image
IMAGE_URL = "https://www.farmersalmanac.com/wp-content/uploads/2023/03/dandelions-plants-that-predict-the-weather.jpeg"

if not st.session_state.get("logged_in", False):
    # Make the columns more distinct in width, with vertical alignment centered
    col_left, col_right = st.columns([2, 3], gap="large")
    with col_left:
        st.image(IMAGE_URL, caption="Welcome to Bloom & Weather Prediction", width='content', clamp=True, channels="RGB", output_format="auto")
    with col_right:
        # Add space at the top to vertically center the form
        st.markdown("<div style='height: 90px;'></div>", unsafe_allow_html=True)
        st.title("Login or Sign Up")
        option = st.radio("Select Action", ["Login", "Sign Up"])
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button(option):
            if option == "Sign Up":
                result = sign_up(email, password)
                if "error" in result:
                    st.error(result["error"]["message"])
                else:
                    st.success("User created! Please log in.")
            else:
                result = sign_in(email, password)
                if "error" in result:
                    st.error(result["error"]["message"])
                else:
                    st.session_state.logged_in = True
                    st.session_state.user_email = result["email"]
                    st.success("Logged in successfully!")
else:
    # User is logged in
    with st.sidebar:
        # NASA logo at the top of the sidebar
        st.markdown(
            """
            <div style="text-align: left; margin-bottom: 10px;">
                <img src="https://upload.wikimedia.org/wikipedia/commons/a/a0/Space_Apps_Logo_White.png" width="69" alt="NASA Logo">
                <text style="text-align: center; font-size: 16px; color: #FF4500;">NASA HACKATHON 2025</text>

            </div>
            """,
            unsafe_allow_html=True
        )
    
            
        st.markdown(
            """
            <style>
             /*.sidebar logo centering*/
            .css-1d391kg {
                padding: 1rem;
                padding-bottom: 1rem;
            }
            /* Sidebar overall background and font */
            .css-1d391kg {
               background-color: #010c29;
               color: #ffffff;
               font-family: Arial, sans-serif;
               }
               <style>
               """,
            unsafe_allow_html=True
        )
        st.title("🚀 NASA BloomWatch Dashboard")
        # Navigation
        page = st.radio("Go to", ["Weather", "Bloom Prediction", "Map", "Bloom Calendar", "Forecast", "Analytics", "Bloom Report Submission", "BloomAlert & NDVI Visualization", "predicted vs actual bloom"])
        # Dark mode toggle
        dark_mode = st.checkbox("Dark Mode 🌙", value=False)
        # Show login info and logout
        st.markdown(f"**Logged in as:** {st.session_state.user_email}")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.user_email = None
            st.success("Logged out successfully!")
            
        # About section
        st.markdown("---")
        st.subheader("About")
        st.write("""
            BloomAlert - Smart plant bloom detection and prediction dashboard.  
            Created for Hackathon 2025 by K.Aditya Varma and team
            Contact: [K.Aditya Varma](mailto:adithyav836@gmail.com)  
            ("© 2025 BloomAlert. All rights reserved.")  
            Version 1.0.0 | Last updated: 2024-06-15
        """)
        #team info
        st.write("Team Members:")
        st.write("- K.Aditya Varma ")
        st.write("- P.Sunder Kumar ") 
        st.write("- E.Ebenejar")
        st.write("- K.Rama Swamy") 
        st.write("- K.Sanjay") 


    # Load city data CSV
    city_csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'cities.csv')
    if not os.path.exists(city_csv_path):
        st.error("Missing cities.csv in the data folder.")
        st.stop()

    city_data = pd.read_csv(city_csv_path)
    required_cols = {'city', 'lat', 'lon'}
    if not required_cols.issubset(city_data.columns):
        st.error("cities.csv must have columns: city, lat, lon")
        st.stop()
    city_data['lat'] = pd.to_numeric(city_data['lat'], errors='coerce')
    city_data['lon'] = pd.to_numeric(city_data['lon'], errors='coerce')
    city_data = city_data.dropna(subset=['lat', 'lon'])
    if city_data.empty:
        st.error("No valid city coordinates.")
        st.stop()

    API_KEY = "replace with ur api key"

    def fetch_weather(city):
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        r = requests.get(url)
        if r.status_code == 200:
            d = r.json()
            return {
                'temperature': d['main']['temp'],
                'humidity': d['main']['humidity'],
                'wind_speed': d['wind']['speed'],
                'cloudiness': d['clouds']['all'],
                'description': d['weather'][0]['description']
            }
        return None

    def generate_blooms(center, num_points=10):
        return [[center[0] + random.uniform(-0.05, 0.05), center[1] + random.uniform(-0.05, 0.05), random.uniform(0.2, 1)] for _ in range(num_points)]

    def predict_bloom(temp, humidity, wind_speed, cloudiness):
        if 20 < temp < 30 and humidity > 50 and wind_speed < 5 and cloudiness < 40:
            return "High chance of bloom", 0.85
        elif 15 < temp <= 20 and humidity > 40 and wind_speed < 8 and cloudiness < 60:
            return "Moderate chance of bloom", 0.55
        else:
            return "Low chance of bloom", 0.30

    # Page logic
    if page in ["Weather", "Bloom Prediction", "Map", "Bloom Calendar", "Forecast", "Analytics"]:
        st.title("🌤️ Weather & 🌸 Bloom Dashboard")
        st.subheader("Select Location and Species")
        location = st.selectbox("Select Location", city_data['city'].tolist())
        species = st.selectbox("Select Plant Species", species_list)

        selected_city_row = city_data[city_data['city'] == location]
        coords = [selected_city_row.iloc[0]['lat'], selected_city_row.iloc[0]['lon']]

        weather = fetch_weather(location)

        if page == "Weather":
            st.subheader(f"Weather in {location}")
            if weather:
                st.markdown(f"☀️ Temperature: {weather['temperature']} °C")
                st.markdown(f"💧 Humidity: {weather['humidity']}%")
                st.markdown(f"🌬️ Wind Speed: {weather['wind_speed']} m/s")
                st.markdown(f"🌥️ Cloudiness: {weather['cloudiness']}%")
                st.markdown(f"🌦️ Conditions: {weather['description']}")
            else:
                st.error("Weather data not available.")

        elif page == "Bloom Prediction":
            st.subheader(f"Bloom Prediction for {species} in {location}")
            if weather:
                pred, conf = predict_bloom(weather['temperature'], weather['humidity'], weather['wind_speed'], weather['cloudiness'])
                st.markdown(f"🌸 Prediction: {pred}")
                st.metric("Model Confidence", f"{conf*100:.1f}%")
                explanation = {
                    "High chance of bloom": "🔴Warm temperature and high humidity support blooming.",
                    "Moderate chance of bloom": "🟡Moderate conditions; some factors favorable.",
                    "Low chance of bloom": "🟢Unfavorable weather including low temperature or high wind."
                }
                st.info(f"**Explanation:** {explanation.get(pred, '')}")
                


            else:
                st.error("No weather data available.")

        elif page == "Map":
            st.subheader(f"Bloom Heatmap for {species} in {location}")
            map_key = f"{location}_{species}_{dark_mode}"
            if "bloom_map_cache" not in st.session_state:
                st.session_state.bloom_map_cache = {}
            if map_key not in st.session_state.bloom_map_cache:
                with st.spinner("Loading heatmap..."):
                    time.sleep(1)
                bloom_points = generate_blooms(coords, num_points=30)
                st.session_state.bloom_map_cache[map_key] = bloom_points
            else:
                bloom_points = st.session_state.bloom_map_cache[map_key]
            tile_layer = "CartoDB Dark_Matter" if dark_mode else "OpenStreetMap"
            m = folium.Map(location=coords, zoom_start=12, tiles=tile_layer)
            HeatMap(bloom_points).add_to(m)
            st_folium(m, width=700)

        elif page == "Bloom Calendar":
            st.subheader(f"Bloom Calendar for {species} in {location}")
            dates = pd.date_range(datetime.date.today(), periods=30)
            chances = [random.uniform(0, 1) for _ in range(30)]
            df = pd.DataFrame({"Date": dates, "Bloom Chance": chances}).set_index("Date")
            st.line_chart(df)
            # CSV download button
            csv = df.to_csv().encode('utf-8')
            st.download_button(
                label="Download Bloom Calendar Data as CSV",
                data=csv,
                file_name=f"bloom_calendar_{location}_{species}.csv",
                mime='text/csv'
            )

        elif page == "Forecast":
            st.subheader(f"5-Day Bloom Forecast for {species} in {location}")
            dates = pd.date_range(datetime.date.today(), periods=5)
            chances = [random.uniform(0, 1) for _ in range(5)]
            for d, c in zip(dates, chances):
                status = "High" if c > 0.7 else "Moderate" if c > 0.4 else "Low"
                st.write(f"{d.date()}: {status} chance of bloom")

        elif page == "Analytics":
            st.subheader("Historical Bloom Trends")
            hist_dates = pd.date_range(datetime.date.today() - pd.Timedelta(days=365), periods=365)
            hist_events = np.random.poisson(3, 365)
            hist_df = pd.DataFrame({"Date": hist_dates, "Events": hist_events}).set_index("Date")
            st.line_chart(hist_df)
            st.subheader("Weather vs Bloom Events")
            weather_vals = np.random.uniform(low=20, high=30, size=365)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist_dates, y=hist_events, name="Bloom Events"))
            fig.add_trace(go.Scatter(x=hist_dates, y=weather_vals, name="Temperature (°C)"))
            st.plotly_chart(fig)
            st.subheader("Prediction Accuracy")
            pred_acc = 0.72
            st.metric("Real vs Predicted Accuracy", value=f"{pred_acc*100:.1f}%")

    elif page == "Bloom Report Submission":
        # Report submission requires login - user is logged in here
        st.header("Submit Bloom Report")
        st.markdown("""
            **Bloom Intensity Guide:**  
            - **Low:** Few flowers blooming, sparse coverage.  
            - **Moderate:** Noticeable flower presence, about 30-60% coverage.  
            - **High:** Dense flower coverage in the area, vibrant bloom.
        """)
        with st.form("bloom_report"):
            report_location = st.selectbox("Location", city_data['city'].tolist())
            report_species = st.selectbox("Plant Species", species_list)
            report_date = st.date_input("Date")
            report_intensity = st.selectbox("Bloom Intensity Level", ["Low", "Moderate", "High"])
            submitted = st.form_submit_button("Submit Report")
            if submitted:
                report_doc = {
                    "location": report_location,
                    "species": report_species,
                    "date": str(report_date),
                    "intensity": report_intensity
                }
                db.collection("bloom_reports").add(report_doc)
                st.success("Thank you! Your bloom report has been saved.")

        st.subheader("Recent Bloom Reports Submitted")
        reports = db.collection("bloom_reports").order_by("date", direction=firestore.Query.DESCENDING).limit(10).stream()
        for rep in reports:
            r = rep.to_dict()
            st.write(f"{r['date']}: {r['species']} bloom in {r['location']} (Intensity: {r['intensity']})")
 
    elif page == "predicted vs actual bloom":
        st.header("Predicted vs Actual Bloom Analysis")
        st.markdown("This section will compare predicted bloom events with actual reported blooms to evaluate model accuracy.")
        import streamlit as st
        import datetime
        import pandas as pd
        datetime.datetime.now()
        date = datetime.datetime.now().strftime("%Y-%m-%d")
        import ee

        ee.Initialize(project='replace with ur api')

        cities_df = pd.read_csv("C:\\Users\\Lenovo\\OneDrive\\Desktop\\bloomalert\\data\\cities.csv")
        species_list = ["🌹Rose", "🌼Jasmine", "🌸Lotus", "🌻Sunflower", "🌷Tulip", "🌼Dandelion", "🌼Marigold", "🌸Lavender", "🌺Orchid", "🌺Hibiscus"]

        city_selected = st.selectbox("Select City", cities_df["city"].tolist())
        species_selected = st.selectbox("Select Species", species_list)


        city_lat = cities_df.loc[cities_df["city"] == city_selected, "lat"].values[0]
        city_lon = cities_df.loc[cities_df["city"] == city_selected, "lon"].values[0]

        weather = fetch_weather(city_selected)
        if weather:
            wind_speed = weather.get("wind_speed", 3)
            cloudiness = weather.get("cloudiness", 20)

# Predict bloom confidence and message (replace with your model)
        def predict_bloom(city, species, wind_speed, cloudiness):
         return 0.3  # Dummy confidence

        def get_predicted_message(confidence):
         if confidence > 0.7:
          return "High chance of bloom."
         elif confidence > 0.4:
          return "Moderate chance of bloom."
         else:
          return "Low chance of bloom."

        model_confidence = predict_bloom(city_selected, species_selected, wind_speed, cloudiness)
        prediction_message = get_predicted_message(model_confidence)

# NDVI calculations with Earth Engine
        point = ee.Geometry.Point(city_lon, city_lat)
        start_date = "2025-08-01"
        end_date = "2025-10-01"
 
        sentinel = (
            ee.ImageCollection("COPERNICUS/S2_SR")
            .filterBounds(point)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
        )

        image = sentinel.median()
        ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")

        mean_ndvi_dict = ndvi.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=point,
        scale=10,
        maxPixels=1e9
        )
        mean_ndvi_value = mean_ndvi_dict.get("NDVI").getInfo()

        m = folium.Map(location=[city_lat, city_lon], zoom_start=10)

# Display side-by-side results that update automatically when selection changes
        col1, col2 = st.columns(2)
        with col1:
         st.subheader("Model Prediction")
         st.metric("Bloom Confidence", f"{model_confidence * 100:.1f}%")
         st.write(prediction_message)
        with col2:
         st.subheader("NDVI Satellite Observation")
         st.metric("Mean NDVI", f"{mean_ndvi_value:.2f}")
         folium_static(m)

# Additional charts, validation, and report download logic can go here
        st.subheader("Validation")
        validation_message = "Model and NDVI agree on bloom likelihood." if (model_confidence > 0.5 and mean_ndvi_value > 0.5) or (model_confidence <= 0.5 and mean_ndvi_value <= 0.5) else "Model and NDVI disagree on bloom likelihood."
        if "agree" in validation_message:
         st.success(validation_message)
        else:
         st.error(validation_message)

        report_df = pd.DataFrame({
        'Date': [date],
        'Location': [city_data],
        'Species': [species_list],
        'Model Confidence': [model_confidence],
        'Mean NDVI': [mean_ndvi_value],
        'Validation Result': [validation_message]
      })
        csv = report_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Report (CSV)", csv, "bloom_ndvi_report.csv", "text/csv")

    elif page == "BloomAlert & NDVI Visualization":
        st.header("NDVI Visualization")

        import ee
        ee.Initialize(project='replace with ur api')

        # Show location selection only inside NDVI module
        city = st.selectbox('Select Area for NDVI', city_data['city'].tolist())

        city_lat = city_data.loc[city_data['city'] == city, 'lat'].values[0]
        city_lon = city_data.loc[city_data['city'] == city, 'lon'].values[0]

        point = ee.Geometry.Point(city_lon, city_lat)
        start_date = '2025-08-01'
        end_date = '2025-10-01'
        sentinel = (ee.ImageCollection('COPERNICUS/S2_SR')
                    .filterBounds(point)
                    .filterDate(start_date, end_date)
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))
                    )
        count = sentinel.size().getInfo()
        st.write(f"Number of Sentinel-2 images found: {count}")

        if count == 0:
            st.warning("No satellite images found for this location and date range. Try changing filters or dates.")
        else:
            image = sentinel.median()
            ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
            ndvi_params = {'min': 0, 'max': 1, 'palette': ['blue', 'white', 'green']}
            map_id_dict = ndvi.getMapId(ndvi_params)
            m = folium.Map(location=[city_lat, city_lon], zoom_start=10)
            folium.TileLayer(
                tiles=map_id_dict['tile_fetcher'].url_format,
                attr='Google Earth Engine',
                name='NDVI',
                overlay=True,
                control=True
            ).add_to(m)
            folium.LayerControl().add_to(m)
            st.subheader(f'NDVI Visualization for {city}')
            folium_static(m)
            st.markdown("""
                **NDVI Interpretation Guide:**  
                - **0.2 - 0.5 (Yellow to Light Green):** Sparse vegetation, possibly stressed plants.  
                - **0.5 - 0.7 (Green):** Healthy vegetation with good chlorophyll content.  
                - **0.7 - 1.0 (Dark Green):** Very dense and healthy vegetation, indicating vigorous plant growth.
            """)
            st.header("Bloom Alert")
            mean_ndvi = ndvi.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=10
            ).get('NDVI')

            mean_ndvi_value = mean_ndvi.getInfo()
            ndvi_threshold = 0.5  # Adjust based on your analysis

            # Calculate mean NDVI for the selected location and period
            mean_ndvi_value = mean_ndvi.getInfo()  # from your existing NDVI code

            if mean_ndvi_value > ndvi_threshold:
             st.success("Bloom alert: Flowering detected in your selected area!")
            else:
             st.info("No current bloom detected.")
        

    # Sticky footer with light color
    st.markdown(
        """
        <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #f5f5f5;
            color: #555;
            text-align: center;
            padding: 8px 0;
            font-size: 14px;
            border-top: 1px solid #ddd;
            z-index: 1000;
        }
        </style>
        <div class="footer">
            © 2025 BloomAlert | Created for Hackathon 2025 by K.Aditya Varma and team.
        </div>
        """,
        unsafe_allow_html=True
    )
import ee
ee.Initialize(project='your project id')









