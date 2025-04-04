import streamlit as st
from streamlit_folium import st_folium
import folium

# Dictionary of cities
cities = {
    "New York": {"lat": 40.7128, "lon": -74.0060, "pop": 8419600},
    "San Francisco": {"lat": 37.7749, "lon": -122.4194, "pop": 883305},
    "Seattle": {"lat": 47.6062, "lon": -122.3321, "pop": 744955},
    "Los Angeles": {"lat": 34.0522, "lon": -118.2437, "pop": 3980400},
    "Houston": {"lat": 29.7604, "lon": -95.3698, "pop": 2328000},
    "Chicago": {"lat": 41.8781, "lon": -87.6298, "pop": 2716000},
}

st.title("Click Markers to Select Cities")

# Set up session_state to store selected cities
if "selected_cities" not in st.session_state:
    st.session_state["selected_cities"] = []

# Create a Folium map centered on the continental US
m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)

# Add markers for each city
for city_name, info in cities.items():
    folium.Marker(
        location=[info["lat"], info["lon"]],
        popup=city_name,                  # The name appears in st_folium's top-level "last_object_clicked_popup"
        tooltip=f"Click for {city_name}", # Displays on hover
    ).add_to(m)

# Render the Folium map in Streamlit
st_map = st_folium(m, width=700, height=500)

# Retrieve the city name from the top-level key in st_map
clicked_city_name = st_map.get("last_object_clicked_popup")

if clicked_city_name:
    # If the clicked city is in our dictionary
    if clicked_city_name in cities:
        # If not already in the session_state, add it
        if clicked_city_name not in st.session_state["selected_cities"]:
            st.session_state["selected_cities"].append(clicked_city_name)
        st.success(f"✅ You selected: {clicked_city_name}")
    else:
        st.warning("⚠️ You clicked on the map but missed a valid marker. Try again.")

st.write("---")
st.subheader("Selected Cities")
if st.session_state["selected_cities"]:
    for city in st.session_state["selected_cities"]:
        info = cities[city]
        st.write(
            f"• **{city}** "
            f"(Population: {info['pop']:,}, "
            f"Coordinates: {info['lat']:.4f}, {info['lon']:.4f})"
        )
else:
    st.info("No cities selected yet. Click a marker on the map to add a city.")
