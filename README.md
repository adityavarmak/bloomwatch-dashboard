# 🌸 BloomWatch Dashboard

Earth observation web app for flowering phenology, built for NASA hackathon.

## Features

- Real-time bloom prediction using weather + satellite (NDVI) data
- Interactive map and NDVI visualization
- Bloom reporting & community validation tools
- Downloadable bloom reports for any location/species
- NASA-branded UI, dark/light mode support

## Folder Structure
bloomalert/
│
├── data/ # Datasets & resources
│ ├── cities.csv
│ └── bloom_map.html
│
├── modeling_visualization/
│ └── app.py # Streamlit dashboard code
|     .streamlit/ # Streamlit config files
|
├── bloom-watch-firebase-admin.js
│
├── requirements.txt # Python dependencies
└── .gitignore

