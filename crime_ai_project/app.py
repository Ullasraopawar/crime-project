from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from sklearn.cluster import KMeans
import joblib

app = Flask(__name__)

data = pd.read_csv("crime_data.csv")
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

# ---------------- HEATMAP ----------------
@app.route("/generate_heatmap")
def generate_heatmap():

    m = folium.Map(
        location=[data.latitude.mean(), data.longitude.mean()],
        zoom_start=12
    )

    heat_data = [
        [row['latitude'], row['longitude'], row['severity']]
        for index, row in data.iterrows()
    ]

    HeatMap(heat_data, radius=15).add_to(m)

    m.save("templates/heatmap.html")
    return render_template("heatmap.html")


# ---------------- CRIME BY HOUR ----------------
@app.route("/crime_by_hour")
def crime_by_hour():

    data['hour'] = pd.to_datetime(data['date_time']).dt.hour
    hourly = data.groupby('hour').size()

    return jsonify(hourly.to_dict())


# ---------------- AREA RANKING ----------------
@app.route("/area_ranking")
def area_ranking():

    ranking = data.groupby('area')['severity'].mean()
    ranking = ranking.sort_values(ascending=False)

    return jsonify(ranking.to_dict())


# ---------------- PATROL ROUTES ----------------
@app.route("/patrol_routes")
def patrol_routes():

    coords = data[['latitude','longitude']]

    kmeans = KMeans(n_clusters=5)
    kmeans.fit(coords)

    centers = kmeans.cluster_centers_

    m = folium.Map(
        location=[data.latitude.mean(), data.longitude.mean()],
        zoom_start=12
    )

    for center in centers:
        folium.Marker(
            location=[center[0], center[1]],
            icon=folium.Icon(color="red"),
            popup="Recommended Patrol Base"
        ).add_to(m)

    m.save("templates/patrol.html")
    return render_template("patrol.html")


# ---------------- PREDICTION ----------------
@app.route("/predict_patrol", methods=["POST"])
def predict_patrol():

    latitude = float(request.form["latitude"])
    longitude = float(request.form["longitude"])
    hour = int(request.form["hour"])
    day = int(request.form["day"])
    month = int(request.form["month"])

    features = np.array([[latitude, longitude, hour, day, month]])
    prediction = model.predict(features)

    if prediction[0] >= 3:
        message = "High Risk Area - Increase Police Patrol 🚔"
    else:
        message = "Low Risk Area"

    return jsonify({"result": message})


if __name__ == "__main__":
    app.run(debug=True)