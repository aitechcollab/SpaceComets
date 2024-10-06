# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 13:11:40 2024

@author: jbalb
"""

import matplotlib.pyplot as plt
import math
import streamlit as st
import pandas as pd
import requests
import plotly.graph_objs as go
import numpy as np
import json
import plotly.express as px
from io import BytesIO
from groq import Groq
import base64
import soundfile as sf
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="AI Interactive All-Inclusive Artistic Orrery of NEOs and Comets", layout="wide")


def deg_to_rad(degrees):
    return math.radians(float(degrees))

def dataset_comets_request():
    
    url = 'https://data.nasa.gov/resource/b67r-rgxc.json'
    
    headers = {
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "accept-encoding": "gzip, deflate, br, zstd",
        "accept-language": "es,es-ES;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
        "cache-control": "max-age=0",
        "cookie": "_frontend_session=WGw2dS95SDdKUDJuM1dBSzE4Rjg5Q3NWcXk4V21ZWVZHbStNZUxjNTVEMFRvMngrMGFuOW9MNDFVL3VMK28xOFJiWHIvMnorZndnTGNSVS9halBrbGVkL21ENEdtbk5LeGo0dktCK3Z5Sk8wS3B5S0JxbHVGZ1ZaU0VFWXRJd2tsblVGdCs5RGdrbDVTaFhqU1hqUUVUd2NDd2RZdHpqdjUwUGwzTmRKYzhCNWhJSmU1aDFqRE12eTNHbFFQYkQ3YnUya093Y0x5S0RldG12Znh2V1ZsellNOXlPKzYyZjlyZ2tjZkpQMndkMGFQaWlPb0F4U0VTMmxUT0FEUW02dnU4enptN0hwMnBkS21RdS9JNXFXYmI5TEZmSThTY3ZFNnB3c3VkQ2c4Y0k9LS14NUlwaFJKemRIU09SVmlVOGxMckVnPT0%3D--a9c2b9585dbf53bb5377279b32f33165014b52fa; _ga=GA1.1.839546742.1728114370; socrata-csrf-token=CDmXGlPTcF2VEJ11+Et1dxINDmQOs/dtFoJ4mTVsnxCnwIf4B7MFGWOivKCsEbFHON7OqSzztg20k7bOt1HhHg==; _ga_CSLL4ZEK4L=GS1.1.1728120815.2.0.1728120815.0.0.0",
        #"if-modified-since": "Wed, 27 Jun 2018 20:49:50 GMT",
        #"if-none-match": "\"Zm94dHJvdC4zMTc1NV8yXzU5R2tkWEVDeVlBemZ0el8yY1g2aF9WWGswYnJB---gzipr20bAc3PaixMhvUWezHWQj0CxXc--gzip--gzip\"",
        #"if-none-match": 'Zm94dHJvdC4zMTc1NV8yXzU5R2tkWEVDeVlBemZ0el8yY1g2aF9WWGswYnJB---gzipr20bAc3PaixMhvUWezHWQj0CxXc--gzip--gzip',
        "priority": "u=0, i",
        "sec-ch-ua": "\"Microsoft Edge\";v=\"129\", \"Not=A?Brand\";v=\"8\", \"Chromium\";v=\"129\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Windows\"",
        "sec-fetch-dest": "document",
        "sec-fetch-mode": "navigate",
        "sec-fetch-site": "none",
        "sec-fetch-user": "?1",
        "upgrade-insecure-requests": "1",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36 Edg/129.0.0.0"
    }
    
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
    except:
        with open("comets_json.json", 'r', encoding='utf-8') as file:
            data = json.load(file)
    
    # Procesar los datos
    cometas_info = []
    for cometa in data:
        # Extraer los valores necesarios
        semieje_mayor = (float(cometa['q_au_1']) + float(cometa['q_au_2'])) / 2  # (q1 + q2) / 2 para obtener el semieje mayor
        excentricidad = float(cometa['e'])
        inclinacion = deg_to_rad(cometa['i_deg'])
        nodo_ascendente = deg_to_rad(cometa['node_deg'])
        argumento_perihelio = deg_to_rad(cometa['w_deg'])
        periodo_orbital = deg_to_rad(cometa['p_yr'])
        moid_au = deg_to_rad(cometa['moid_au'])
    
        # Agregar los datos procesados a la lista
        cometas_info.append({
            'Objeto': cometa['object_name'],
            'Semieje Mayor (AU)': semieje_mayor,
            'Excentricidad': excentricidad,
            'Inclinación (rad)': inclinacion,
            'Nodo Ascendente (rad)': nodo_ascendente,
            'Argumento del Perihelio (rad)': argumento_perihelio,
            'Periodo Orbital': periodo_orbital,
            'moid_au':moid_au
        })

    # Convertir los datos a un DataFrame de pandas
    df_cometas = pd.DataFrame(cometas_info)
    return df_cometas

# Función para resolver la ecuación de Kepler (anomalía excéntrica E)
def solve_kepler_comets(M, e, tol=1e-6):
    E = M
    while True:
        delta = E - e * np.sin(E) - M
        if np.abs(delta) < tol:
            break
        E = E - delta / (1 - e * np.cos(E))
    return E

# Función para trazar la órbita 3D del cometa utilizando Plotly
def plot_orbit_3d_comet(semieje_mayor, excentricidad, inclinacion, nodo_ascendente, argumento_perihelio, periodo_orbital, moid_au):

    M_values = np.linspace(0, 2 * np.pi, 1000)
    E_values = np.array([solve_kepler_comets(M, excentricidad) for M in M_values])

    # Coordenadas en el plano orbital
    r = semieje_mayor * (1 - excentricidad * np.cos(E_values))
    x_orbit = r * np.cos(E_values)
    y_orbit = r * np.sin(E_values)

    # Transformación a 3D (desde el plano orbital a coordenadas heliocéntricas)
    x_helio = x_orbit * (np.cos(nodo_ascendente) * np.cos(argumento_perihelio) - np.sin(nodo_ascendente) * np.sin(argumento_perihelio) * np.cos(inclinacion)) - \
              y_orbit * (np.cos(nodo_ascendente) * np.sin(argumento_perihelio) + np.sin(nodo_ascendente) * np.cos(argumento_perihelio) * np.cos(inclinacion))

    y_helio = x_orbit * (np.sin(nodo_ascendente) * np.cos(argumento_perihelio) + np.cos(nodo_ascendente) * np.sin(argumento_perihelio) * np.cos(inclinacion)) + \
              y_orbit * (np.cos(argumento_perihelio) * np.sin(nodo_ascendente) - np.cos(nodo_ascendente) * np.sin(argumento_perihelio) * np.cos(inclinacion))

    z_helio = x_orbit * (np.sin(inclinacion) * np.sin(argumento_perihelio)) + y_orbit * (np.sin(inclinacion) * np.cos(argumento_perihelio))

    # Crear la figura 3D con Plotly
    fig = go.Figure()

    # Agregar la órbita del cometa
    fig.add_trace(go.Scatter3d(
        x=x_helio, y=y_helio, z=z_helio,
        mode='lines',
        line=dict(color='orange', width=2),
        name='Orbit of the Comet'
    ))

    # Agregar el Sol en el centro
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(color='yellow', size=10),
        name='Sol'
    ))

    # Configuración de los ejes
    fig.update_layout(
        scene=dict(
            xaxis_title='X (AU)',
            yaxis_title='Y (AU)',
            zaxis_title='Z (AU)',
        ),
        title='3D Comet Orbit',
        showlegend=True,
        #width=100,  
        height=500
    )
    
    theta = np.linspace(0, 2 * np.pi * (semieje_mayor/10), 1000)
    radii = (1 + excentricidad * np.cos(theta)) * (1 - excentricidad) * (1 + 0.1 * np.random.rand(1000))

    plt.figure(figsize=(10, 10))
    plt.style.use('dark_background')
    
    # Create a colormap based on the radii
    colors = plt.cm.viridis(radii / max(radii))
    
    x_disp = x_helio + np.random.normal(0, moid_au+0.1, x_helio.shape)
    y_disp = y_helio + np.random.normal(0, moid_au+0.1, y_helio.shape)
    
    # Scatter plot to create the abstract pattern
    plt.scatter(x_disp, y_disp, c=colors, s=10, alpha=0.6)
    plt.title('Abstract Art Inspired by Aroa Borrego & Javier Balbas', fontsize=10, color='white')
    plt.axis('off')

    # Save the plot to a BytesIO object
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight', facecolor='black')
    plt.close()  # Close the plot to avoid display
    img_buf.seek(0)  
    
    def normalize(data, min_val, max_val):
        normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
        return normalized_data * (max_val - min_val) + min_val
    
    def generate_sine_wave(frequency, duration, sample_rate=44100):
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        sine_wave = np.sin(2 * np.pi * frequency * t)
        return sine_wave
    
    x_frequencies = normalize(x_disp, 300, 3000)  # Frecuencias basadas en X
    y_frequencies = normalize(y_disp, 300, 3000)  # Frecuencias basadas en Y
    
    duration = 0.05  
    audio = np.array([])
    
    for x_freq, y_freq in zip(x_frequencies, y_frequencies):
        # Para cada punto de datos, crear un sonido con la frecuencia combinada de X e Y
        combined_wave = generate_sine_wave(x_freq, duration) + generate_sine_wave(y_freq, duration)
        audio = np.concatenate((audio, combined_wave))
        
    return img_buf, fig, audio


def total_pages(api_key):
    url_total_pages = f"https://api.nasa.gov/neo/rest/v1/neo/browse?api_key={api_key}"
    res_tp = requests.get(url_total_pages)
    if res_tp.status_code != 200:
        st.error("Error API NEOs: {}".format(res_tp.status_code))
        return []
    data = res_tp.json()
    return data['page']['total_pages']

def extract_and_filter_neo_data(api_key, total_pages, hazardous_only, size):
    records = []
    
    with st.spinner('Calculating Kepler equations and conversion to orbital parameters...'):

        for i in range(1, int(total_pages/size) + 1):
            
            url = f"https://api.nasa.gov/neo/rest/v1/neo/browse?api_key={api_key}&size={size}&page={i}"
            
            response = requests.get(url)
           
            if response.status_code != 200:
                st.error(f"Error pag. {i}. Code: {response.status_code}")
                break
           
            neo_data = response.json()['near_earth_objects']
           
            # Filtrar y procesar datos en la misma función
            for neo in neo_data:
                if hazardous_only == True:
                    if neo['is_potentially_hazardous_asteroid']== True:
                        records.append({
                            'name': neo['name'],
                            'id': neo['id'],
                            'magnitude': neo['estimated_diameter']['kilometers']['estimated_diameter_max'],
                            'close_approach_date': neo['close_approach_data'][0]['close_approach_date'],
                            'miss_distance': neo['close_approach_data'][0]['miss_distance']['kilometers'],
                            'orbiting_body': neo['close_approach_data'][0]['orbiting_body'],
                            'is_hazardous': neo['is_potentially_hazardous_asteroid'],
                            'nasa_jpl_url': neo['nasa_jpl_url'],
                            'orbit_a': float(neo['orbital_data']['semi_major_axis']),
                            'orbit_e': float(neo['orbital_data']['eccentricity']),
                            'orbit_i': float(neo['orbital_data']['inclination']),
                            'orbit_omega': float(neo['orbital_data']['ascending_node_longitude']),
                            'orbit_w': float(neo['orbital_data']['perihelion_argument']),
                            'orbit_ma': float(neo['orbital_data']['mean_anomaly']),
                        })
                else:
                    records.append({
                        'name': neo['name'],
                        'id': neo['id'],
                        'magnitude': neo['estimated_diameter']['kilometers']['estimated_diameter_max'],
                        'close_approach_date': neo['close_approach_data'][0]['close_approach_date'],
                        'miss_distance': neo['close_approach_data'][0]['miss_distance']['kilometers'],
                        'orbiting_body': neo['close_approach_data'][0]['orbiting_body'],
                        'is_hazardous': neo['is_potentially_hazardous_asteroid'],
                        'nasa_jpl_url': neo['nasa_jpl_url'],
                        'orbit_a': float(neo['orbital_data']['semi_major_axis']),
                        'orbit_e': float(neo['orbital_data']['eccentricity']),
                        'orbit_i': float(neo['orbital_data']['inclination']),
                        'orbit_omega': float(neo['orbital_data']['ascending_node_longitude']),
                        'orbit_w': float(neo['orbital_data']['perihelion_argument']),
                        'orbit_ma': float(neo['orbital_data']['mean_anomaly']),
                    })
           
       
        if records:
            return pd.DataFrame(records)
        else:
            st.warning("No Dangerous NEOs.")
            return pd.DataFrame()

def create_orrery_3d_neo(neo_df):
    fig = go.Figure()

    # Añadir el Sol
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=20, color='yellow', symbol='circle'),
        name='Sol'
    ))
   
    with st.spinner('Calculating Kepler equations and conversion to orbital parameters...'):
       
        # Añadir las órbitas de los NEOs
        for index, row in neo_df.iterrows():
            if row is not None:
                
                # Extraer y convertir los parámetros orbitales
                a = float(row['orbit_a'])  # Semieje mayor en AU
                e = float(row['orbit_e'])  # Excentricidad
                i = np.radians(float(row['orbit_i']))  # Inclinación en radianes
                omega = np.radians(float(row['orbit_omega']))  # Nodo ascendente en radianes
                w = np.radians(float(row['orbit_w']))  # Argumento del perihelio en radianes
                M = np.radians(float(row['orbit_ma']))  # Anomalía media en radianes
               
                # Resolver la ecuación de Kepler para obtener la anomalía excéntrica
                E = solve_kepler(M, e)
   
                # Obtener la anomalía verdadera (v)
                v = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))
               
                # Obtener la distancia al sol (r)
                r = a * (1 - e * np.cos(E))
               
                # Coordenadas en el plano orbital
                x_orbital = r * np.cos(v)
                y_orbital = r * np.sin(v)
   
                # Transformación a coordenadas heliocéntricas
                x_heliocentric = (np.cos(omega) * np.cos(w + v) - np.sin(omega) * np.sin(w + v) * np.cos(i)) * r
                y_heliocentric = (np.sin(omega) * np.cos(w + v) + np.cos(omega) * np.sin(w + v) * np.cos(i)) * r
                z_heliocentric = (np.sin(w + v) * np.sin(i)) * r
   
                # Añadir órbita
                theta = np.linspace(0, 2 * np.pi, 100)
                r_orbit = a * (1 - e**2) / (1 + e * np.cos(theta))
                x_orbit_heliocentric = (np.cos(omega) * np.cos(w + theta) - np.sin(omega) * np.sin(w + theta) * np.cos(i)) * r_orbit
                y_orbit_heliocentric = (np.sin(omega) * np.cos(w + theta) + np.cos(omega) * np.sin(w + theta) * np.cos(i)) * r_orbit
                z_orbit_heliocentric = (np.sin(w + theta) * np.sin(i)) * r_orbit
   
                # Añadir la órbita al gráfico
                fig.add_trace(go.Scatter3d(
                    x=x_orbit_heliocentric,
                    y=y_orbit_heliocentric,
                    z=z_orbit_heliocentric,
                    mode='lines',
                    name=row['name'] + ' Órbita'
                ))
   
                # Añadir posición del NEO
                fig.add_trace(go.Scatter3d(
                    x=[x_heliocentric],
                    y=[y_heliocentric],
                    z=[z_heliocentric],
                    mode='markers',
                    marker=dict(size=5, color='red' if row['is_hazardous'] else 'blue'),
                    name=row['name'] + (' (PHA)' if row['is_hazardous'] else '')
                ))
   
        # Configuración de la figura 3D
        fig.update_layout(
            scene=dict(
                xaxis_title='X (AU)',
                yaxis_title='Y (AU)',
                zaxis_title='Z (AU)',
                aspectmode='data'
            ),
            title='Orrery of Near Earth Orbits (NEOs) :rocket:',
            showlegend=True
        )
   
    return fig

# Función para resolver la ecuación de Kepler y obtener la anomalía excéntrica
def solve_kepler(M, e, tolerance=1e-6):
    E = M  # E es la anomalía excéntrica
    while True:
        delta_E = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E -= delta_E
        if np.abs(delta_E) < tolerance:
            break
    return E

def image_to_text(image_path, cometa_selecionado):
    
    def encode_image(image):
        if isinstance(image, str):
            with open(image, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        elif isinstance(image, BytesIO):
            return base64.b64encode(image.getvalue()).decode('utf-8')
        else:
            raise ValueError("Tipo de entrada no soportado, debe ser str o BytesIO")

    GROQ_API_KEY = os.getenv('GROQ_API_KEY')

    # Getting the base64 string
    base64_image = encode_image(image_path)
    
    client = Groq(api_key=GROQ_API_KEY)
    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Describe the image of an abstract representation of the orbit of {cometa_selecionado}. The description should combine scientific and technical elements, as well as a romantic touch that illustrates the beauty of art and science. Imagine you are speaking to someone who cannot see the image, so be sure to use evocative language that allows them to visualize the constellation in their mind. Start by describing the arrangement of the stars in the constellation, the colors and shapes of the abstract art surrounding it, and how these representations may reflect the real characteristics of the stars and their context in the universe. Additionally, include a reflection on the connection between art and science, and how the abstract representation of the constellation can evoke feelings of wonder and curiosity about the cosmos."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        model="llava-v1.5-7b-4096-preview",
    )
    
    text_imagen = chat_completion.choices[0].message.content
    return text_imagen

def llm_cometas_y_neos(pregunta, selected_neo_comet):
    if pregunta:

        GROQ_API_KEY = os.getenv('GROQ_API_KEY')

        groq_url = "https://api.groq.com/openai/v1/chat/completions"

        headers_groq = {
            "Accept-Language": "en-VN;q=1.0",
            "User-Agent": "MindMac/1.9.11 (app.mindmac.macos; build:80; macOS 14.4.1) Alamofire/5.8.0",
            "Accept": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        payload_groq = {
            "temperature": 0.1,
            #"stop": ["<|im_end|>"],
            #"n": 1,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert in astronomy, satellites, NEOs, LEOs, comets, planets, stars, solar systems, everything. You know everything about NASA."
                },
                {
                    "role": "user",
                    "content": f"Respond about the NEO {selected_neo_comet} as if you were {selected_neo_comet} to the following question: {pregunta}",
                }
            ],
            #"stream": False,
            #"max_tokens": 1024,
            "model": "llama3-70b-8192"
            #"model": "llama-3.1-70b-versatile"
        }
       
        response = requests.post(groq_url, headers=headers_groq, data=json.dumps(payload_groq))
        return response
    else:
        st.warning("Por favor, escribe una pregunta antes de enviar.")
    
        
tab_neos, tab_comets = st.tabs(["NEOs", "Comets"])

api_key = os.getenv('NASA_API_KEY')

neo_names = None
    
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .stDeployButton {
            visibility: hidden;
            }
            div[data-testid="stStatusWidget"] div button {
            display: none;
            }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
size = 19
df_cometas = dataset_comets_request()

with st.sidebar:
    st.header("NEO Configuration")
    total_pags = total_pages("laTpK3JUEof9QuZHa8e2cE6GOcVfso4oDHgIvP0c")
    pages_to_analyze = st.select_slider(
        'Select how many pages to analyze:',
        #options=list(range(1, total_pags + 1)),
        options = list(range(size, (total_pags + 1) * size, size)),
        value=size
    )
   
    # Checkbox para filtrar solo NEOs peligrosos
    hazardous_only = st.toggle('Select only potentially dangerous NEOs')
   
    # Botón para ejecutar el análisis
    run_analysis_neos = st.button('Run NEOs Analysis')
    
    st.header("Comet Configuration")
    cometa_seleccionado = st.selectbox("Select the name of the comet", df_cometas['Objeto'])

    run_analysis_comets = st.button('Run Comet Analysis')
    
    st.info("AI-Interactive-Artistic-All-Inclusive Overry APP by Aroa & Javier")

with tab_neos:    
    st.title("Near-Earth Object (NEO) Orrery :earth_africa: :ringed_planet:")

    # Ejecutar análisis al pulsar el botón
    if run_analysis_neos:
        print(hazardous_only)
        progress_text = "Data Extraction in Progress..."
        records = []
        neo_df = extract_and_filter_neo_data(api_key, pages_to_analyze, hazardous_only, size)
       
        if not neo_df.empty:
            
            col1, col2 = st.columns([1, 2])  # proporciones 1:2
       
            with col1:
                st.write("Information about NEOs:")
                # Opción para filtrar por NEOs peligrosos
       
                st.dataframe(neo_df)
       
            with col2:
                fig = px.scatter(neo_df, x='close_approach_date', y='magnitude',
                                 color='is_hazardous',
                                 hover_name='name',
                                 title="Chart of NEOs by Date of Approach and Magnitude")
                st.plotly_chart(fig)
               
            orrery_3d = create_orrery_3d_neo(neo_df)
           
            st.plotly_chart(orrery_3d)
       
           
            neo_names = neo_df['name'].tolist()
            st.session_state.neo_names = neo_df['name'].tolist()  
           
    if 'neo_names' in st.session_state:
       
        neo_names = st.session_state.neo_names
        selected_neo = st.selectbox("Select a NEO:", neo_names)
    
        pregunta = st.text_input("Write your question about the selected NEO:")
       
        if st.button("Ask IA :brain:"):
            with st.spinner('Running Query Model with Artificial Intelligence...'):
                response_groq = llm_cometas_y_neos(pregunta, selected_neo)
                if response_groq.status_code == 200:
                    data_groq = json.loads(response_groq.text)
                    respuesta_groq = data_groq["choices"][0]["message"]["content"]
                    st.success("Answer :robot_face: {}".format(respuesta_groq))
                else:
                    st.error(response_groq.text)
                    st.error("Código de error: {}".format(response_groq.status_code))
            

    else:
        st.info("Run NEO analysis to visualize data.")
        
with tab_comets:        
    st.title("Near-Earth Comets :earth_africa: :comet:")
    if run_analysis_comets:
        with st.spinner(f"Generating the comet analysis and running the AI for {cometa_seleccionado}..."):
            col11, col22 = st.columns(2)
            
            cometa_info = df_cometas[df_cometas['Objeto'] == cometa_seleccionado].iloc[0]
            st.session_state.cometa_seleccionado = cometa_seleccionado
            
            image_comet,plot_fig_comet,audio_comet= plot_orbit_3d_comet(
                semieje_mayor=cometa_info['Semieje Mayor (AU)'],
                excentricidad=cometa_info['Excentricidad'],
                inclinacion=cometa_info['Inclinación (rad)'],
                nodo_ascendente=cometa_info['Nodo Ascendente (rad)'],
                argumento_perihelio=cometa_info['Argumento del Perihelio (rad)'],
                periodo_orbital=cometa_info['Periodo Orbital'],
                moid_au=cometa_info['moid_au']
            )
            with col11:
                st.image(image_comet)
                
    
            with col22:
                st.plotly_chart(plot_fig_comet)
            
            st.subheader(f"_Melody_ that represents _The Orbit Artistic Work_ of :blue[{cometa_seleccionado} ] :art: :loud_sound:")
            st.audio(audio_comet, sample_rate=44100)
            st.subheader("AI Analysis :brain: :robot_face:")
            text_orbita_img = image_to_text(image_comet, cometa_seleccionado)
            st.success(":robot_face: {}".format(text_orbita_img))
            
            api_key_eleven_labs = os.getenv('EL_API_KEY')
            voice_id = "9BWtsMINqrJLrRacOk9x"
            
            url_el = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            headers_el = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": api_key_eleven_labs
            }
            
            data = {
                "text": text_orbita_img,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75
                }
            }
            """ """
            with st.spinner("Generating Audio with Artificial Intelligence..."):
         
                response_el = requests.post(url_el, headers=headers_el, json=data)
                if response_el.status_code == 200:
                    audio_path = "Podcast_Generado.mp3"
                    with open(audio_path, "wb") as audio_file:
                        audio_file.write(response_el.content)
                    st.audio(audio_path, format="audio/mpeg", start_time=0)
                else:
                    print(f"Error: {response_el.status_code}")
                    print(response_el.text)
                    st.warning("Error Audio: " + str(response_el.status_code))
            
            