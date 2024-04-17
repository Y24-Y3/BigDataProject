from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import folium
app = Flask(__name__)

df = pd.read_csv('Motor_Vehicle_Collisions_-_Crashes.csv', encoding='latin1', low_memory=False)




@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/map')
def map():
    accident_map = folium.Map(location=[df['LATITUDE'].mean(), df['LONGITUDE'].mean()], zoom_start=10)
    accident_map.save('templates/accident_map.html')

    return render_template('home.html')

if __name__ == '__main__':
    app.run()
    