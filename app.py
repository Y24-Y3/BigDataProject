from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import folium
import textwrap
import plotly.express as px
import json
import plotly
import plotly.graph_objs as go
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from folium import plugins
from folium.plugins import MarkerCluster, HeatMap
from statsmodels.tsa.arima.model import ARIMA
app = Flask(__name__)

df = pd.read_csv('Motor_Vehicle_Collisions_-_Crashes.csv', encoding='latin1', low_memory=False)

# drop rows with missing lat and long values
df = df.dropna(subset=['LATITUDE', 'LONGITUDE', 'LOCATION'])

#===============================================
# time Series route
#===============================================

model = pickle.load(open('models/model.pkl', 'rb'))

@app.route('/time_series')
def time_series():
    affected()
    injured()
    killed()
    brook_predict()
    queens_predict()
    bronx_predict()
    manhatt_predict()
    staten_predict()
    model_summaries = summary()


    return render_template('time_series.html', model_summaries=model_summaries)


""" @app.route('/time_series/model_summary')
def model_summary():
    model_summaries = summary()
    return render_template('model_summary.html', model_summaries=model_summaries)
 """

# ====================================Readingf Csv ===========================

brookdf = pd.read_csv("brooklyn.csv")
queendf = pd.read_csv("queens.csv")
bronxdf = pd.read_csv("bronx.csv")
mandf = pd.read_csv("manhattan.csv")
statdf = pd.read_csv("staten_island.csv")

# ============================= Analysis ===================================

#@app.route('/time_series/affected')
def affected():
    brookdf['num_affected(thousands)'].plot(figsize=(10,5), linewidth=3, fontsize=15)
    queendf['num_affected(thousands)'].plot(figsize=(10,5), linewidth=3, fontsize=15)
    bronxdf['num_affected(thousands)'].plot(figsize=(10,5), linewidth=3, fontsize=15)
    mandf['num_affected(thousands)'].plot(figsize=(10,5), linewidth=3, fontsize=15)
    statdf['num_affected(thousands)'].plot(figsize=(10,5), linewidth=3, fontsize=15)
    plt.xlabel('Year', fontsize=15);
    plt.ylabel('Number Affected(thousands)', fontsize=15);
    plt.legend(('Brooklyn','Queens','Bronx','Manhattan','Staten Island'))
    plt.title("Affected Persons by Borough")

    plt.savefig('static/affected.png')

def injured():
    brookdf['num_injured(thousands)'].plot(figsize=(10,5), linewidth=3, fontsize=15)
    queendf['num_injured(thousands)'].plot(figsize=(10,5), linewidth=3, fontsize=15)
    bronxdf['num_injured(thousands)'].plot(figsize=(10,5), linewidth=3, fontsize=15)
    mandf['num_injured(thousands)'].plot(figsize=(10,5), linewidth=3, fontsize=15)
    statdf['num_injured(thousands)'].plot(figsize=(10,5), linewidth=3, fontsize=15)
    plt.xlabel('Year', fontsize=15);
    plt.ylabel('Number Injured(thousands)', fontsize=15);
    plt.legend(('Brooklyn','Queens','Bronx','Manhattan','Staten Island'))
    plt.title("Injured Persons by Borough")

    plt.savefig('static/injured.png')



def killed():
    brookdf['num_killed(tens)'].plot(figsize=(10,5), linewidth=3, fontsize=15)
    queendf['num_killed(tens)'].plot(figsize=(10,5), linewidth=3, fontsize=15)
    bronxdf['num_killed(tens)'].plot(figsize=(10,5), linewidth=3, fontsize=15)
    mandf['num_killed(tens)'].plot(figsize=(10,5), linewidth=3, fontsize=15)
    statdf['num_killed(tens)'].plot(figsize=(10,5), linewidth=3, fontsize=15)
    plt.xlabel('Year', fontsize=15);
    plt.ylabel('Number Killed(tens)', fontsize=15);
    plt.legend(('Brooklyn','Queens','Bronx','Manhattan','Staten Island'))
    plt.title("Killed Persons by Borough")

    plt.savefig('static/killed.png')

#======================Prediction Model ===========================

def brook_predict():
    plt.plot(model.predict(dynamic=False))
    plt.plot(brookdf['num_affected(thousands)'])
    plt.legend(("Prediction","Actual"))
    plt.title("Predicted vs Actual Number of Affected in Brooklyn")

    plt.savefig('static/brook_predict.png')

def queens_predict():
    plt.plot(model.predict(dynamic=False))
    plt.plot(queendf['num_affected(thousands)'])
    plt.legend(("Prediction","Actual"))
    plt.title("Predicted vs Actual Number of Affected in Queens")

    plt.savefig('static/queens_predict.png')

def bronx_predict():
    plt.plot(model.predict(dynamic=False))
    plt.plot(bronxdf['num_affected(thousands)'])
    plt.legend(("Prediction","Actual"))
    plt.title("Predicted vs Actual Number of Affected in Bronx")

    plt.savefig('static/bronx_predict.png')

def manhatt_predict():
    plt.plot(model.predict(dynamic=False))
    plt.plot(mandf['num_affected(thousands)'])
    plt.legend(("Prediction","Actual"))
    plt.title("Predicted vs Actual Number of Affected in Manhattan")

    plt.savefig('static/manhattan_predict.png')

def staten_predict():
    plt.plot(model.predict(dynamic=False))
    plt.plot(statdf['num_affected(thousands)'])
    plt.legend(("Prediction","Actual"))
    plt.title("Predicted vs Actual Number of Affected in Staten Island")

    plt.savefig('static/staten_predict.png')


#====================== Summary ===========================
""" 
def summary():
    arima_modelq = ARIMA(queendf['num_affected(thousands)'],order=(1,1,3))
    arima_modelq_fit = arima_modelq.fit()

    arima_modelbr = ARIMA(brookdf['num_affected(thousands)'],order=(1,1,3))
    arima_modelbr_fit = arima_modelbr.fit()

    arima_modelbx = ARIMA(bronxdf['num_affected(thousands)'],order=(1,1,3))
    arima_modelbx_fit = arima_modelbx.fit()

    arima_modelm = ARIMA(mandf['num_affected(thousands)'],order=(1,1,3))
    arima_modelm_fit = arima_modelm.fit()

    arima_modelst = ARIMA(statdf['num_affected(thousands)'],order=(1,1,3))
    arima_modelst_fit = arima_modelst.fit()
 """

def summary():
    # Assuming ARIMA and other necessary imports are done
    models = {
        'Queens': ARIMA(queendf['num_affected(thousands)'], order=(1, 1, 3)).fit(),
        'Brooklyn': ARIMA(brookdf['num_affected(thousands)'], order=(1, 1, 3)).fit(),
        'Bronx': ARIMA(bronxdf['num_affected(thousands)'], order=(1, 1, 3)).fit(),
        'Manhattan': ARIMA(mandf['num_affected(thousands)'], order=(1, 1, 3)).fit(),
        'Staten Island': ARIMA(statdf['num_affected(thousands)'], order=(1, 1, 3)).fit()
    }
    
    model_summaries = {}
    for borough, model in models.items():
        summary_info = {
            'Dep. Variable': model.model.endog_names,
            'No. Observations': len(model.model.endog),
            'Model': model.model.order,
            'Log Likelihood': model.llf,
            'AIC': model.aic,
            'BIC': model.bic,
            'HQIC': model.hqic,
            'Covariance Type': model.cov_type,
            'Coefficients': model.params.tolist(),
            'Standard Errors': model.bse.tolist(),
            'z-scores': model.tvalues.tolist(),
            'P>|z|': model.pvalues.tolist(),
            'Confidence Intervals': model.conf_int().values.tolist()
        }
        model_summaries[borough] = summary_info
    
    return model_summaries




# Maps
#================================================

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/map')
def map():
    return render_template('home.html')

@app.route('/map/nyc_map')
def nyc_map():
    map = folium.Map(location=[40.7128, -74.0060], zoom_start=10)
    map.save('templates/accident_map.html')

    return render_template('accident_map.html')

@app.route('/map/heatmap')
def heatmap():
    accident_map = folium.Map(location=[df['LATITUDE'].mean(), df['LONGITUDE'].mean()], zoom_start=10)
    heat_data = [[row['LATITUDE'], row['LONGITUDE']] for index, row in df.iterrows()]
    folium.plugins.HeatMap(heat_data).add_to(accident_map)
    accident_map.save('templates/heatmap.html')

    return render_template('heatmap.html')

@app.route('/map/cluster')
def cluster():
    map = folium.Map(location=[df['LATITUDE'].mean(), df['LONGITUDE'].mean()], zoom_start=10)

    grouped = df.groupby('BOROUGH')

    for name, group in grouped:
        marker_cluster = MarkerCluster(name=name).add_to(map)
        for index, row in group.iterrows():
            location = [row['LATITUDE'], row['LONGITUDE']]
            popup = folium.Popup(f"Borough: {name}", max_width=200)
            folium.Marker(location, popup=popup).add_to(marker_cluster)

    # Add LayerControl to the map
    folium.LayerControl().add_to(map)

    map.save('templates/cluster_map.html')

    return render_template('cluster_map.html')

@app.route('/map/injury_map')
def injury_map():
    map_accidents = folium.Map(location=[df['LATITUDE'].mean(), df['LONGITUDE'].mean()], zoom_start=10)

    # Define a function to scale the size of the markers based on the number of persons injured
    def scale_size(persons_injured):
        return persons_injured * 10

    for i in range(0, len(df), 1000):
        row = df.iloc[i]
        location = [row['LATITUDE'], row['LONGITUDE']]
        persons_injured = row['NUMBER OF PERSONS INJURED']
        legend = f'<strong>Vehicle:</strong> {row["VEHICLE TYPE CODE 1"]} <br><strong> Persons Injured:</strong> {persons_injured}<strong>  <br> Persons Killed: {row["NUMBER OF PERSONS KILLED"]} </strong> <br> <strong> Accident Factors: </strong> <li> {row["CONTRIBUTING FACTOR VEHICLE 1"]} <li> {row["CONTRIBUTING FACTOR VEHICLE 2"]}'
        size = scale_size(persons_injured)
        popup = folium.Popup(legend, max_width=200)
        icon = folium.Icon(color='red', icon='info-sign', prefix='glyphicon')
        folium.Marker(location, popup=popup, icon=icon).add_to(map_accidents)

    map_accidents.save('templates/injury_map.html')

    return render_template('injury_map.html')

@app.route('/map/clusters')
def clusters():
    df_sample = df.sample(n=10000, random_state=42)

    # Create a map centered around the mean coordinates of the sampled accidents
    map = folium.Map(location=[df_sample['LATITUDE'].mean(), df_sample['LONGITUDE'].mean()], zoom_start=10)

    # Group the sampled DataFrame by borough
    grouped = df_sample.groupby('BOROUGH')

    # Create a MarkerCluster for each borough
    for name, group in grouped:
        marker_cluster = MarkerCluster(name=name).add_to(map)
        for index, row in group.iterrows():
            location = [row['LATITUDE'], row['LONGITUDE']]
            popup = folium.Popup(f"Borough: {name}", max_width=200)
            folium.Marker(location, popup=popup).add_to(marker_cluster)

    # Add LayerControl to the map
    folium.LayerControl().add_to(map)

    # Display the map
    map.save('templates/cluster_by_borough_limited.html')

    return render_template('cluster_by_borough_limited.html')

#=====================================================
#                        EDA
#=====================================================







if __name__ == '__main__':
    app.run()
    