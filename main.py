import json
import ast
import pandas as pd
import numpy as np
from fastapi import FastAPI
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from model import load_model, make_predictions
import pickle

#Comienzo de la api
#para levantar fast api: uvicorn main:app --reload
app = FastAPI()

#lectura del json y creación data frame
rows = []
with open('steam_games.json') as f: 
    rows.extend(ast.literal_eval(line) for line in f)
df = pd.DataFrame(rows)


#Limpieza de data
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
specific_date = pd.to_datetime('1900-01-01')
df['release_date'] = df['release_date'].fillna(specific_date)
df['metascore'] = pd.to_numeric(df['metascore'], errors='coerce')
df['price'] = pd.to_numeric(df['price'], errors='coerce')

replacement_values = {
    'publisher': '',
    'genres': '',
    'tags': '',
    'discount_price': 0,     
    'price': 0,
    'specs': '',
    'reviews_url': '',            
    'metascore': 0,         
    'app_name': '',        
    'title': '', 
    'id': '',
    'sentiment': '',
    'developer': ''            

}
df.fillna(value=replacement_values, inplace=True)

# Retorna los 5 géneros más vendidos en el año indicado
@app.get('/genero/')
def genero(año: int):
    filtered_df = df[df['release_date'].dt.year == año]
    # desanidar
    exploded_genres_df = filtered_df.explode('genres')
    top_genres = exploded_genres_df['genres'].value_counts().nlargest(5).index.tolist()
    return top_genres

# Retorna juegos lanzados en el año indicado
@app.get('/juegos/')
def juegos(año: int):
    filtered_df = df[df['release_date'].dt.year == año]
    released_games = filtered_df['app_name'].tolist()
    return released_games

# Retorna 5 specs más repetidos en el año indicado
@app.get('/specs/')
def specs(año: int):
    filtered_df = df[df['release_date'].dt.year == año]
    exploded_specs_df = filtered_df.explode('specs')
    top_specs = exploded_specs_df['specs'].value_counts().nlargest(5).index.tolist()
    return top_specs

# Retorna cantidad de juegos lanzados con early acces en el año indicado
@app.get('/earlyacces/')
def earlyacces(año: int):
    filtered_df = df[df['release_date'].dt.year == año]
    count_early_access = len(filtered_df[filtered_df['early_access'] == True])
    return count_early_access

# Retorna lista con registros categorizados con un "sentiment" específico, en el año indicado
@app.get('/sentiment/')
def sentiment(año: int):
    filtered_df = df[df['release_date'].dt.year == año]
    sentiment_counts = filtered_df['sentiment'].value_counts().to_dict()
    return sentiment_counts

# Retorna los 5 juegos con mayor metascore en el año indicado
@app.get('/metascore/')
def metascore(año: int):
    filtered_df = df[df['release_date'].dt.year == año]
    top_metascore_games = filtered_df.nlargest(5, 'metascore')[['app_name', 'metascore']].set_index('app_name').to_dict()['metascore']
    return top_metascore_games

class Item(BaseModel):
    genero: str
    año: int
    metascore: int

app = FastAPI()

model = pickle.load(open('predictions.pkl', 'rb'))

@app.post("/prediccion/")
async def create_prediccion(item: Item):
    # Convertir 'genero' a números (usando one-hot encoding)
    # Tendrías que tener una lista con todos los géneros posibles de tus datos de entrenamiento
    genres = ['genre1', 'genre2', 'genre3', ..., 'genreN']
    genre_data = [1 if item.genero == genre else 0 for genre in genres]
    data = [item.año, item.metascore] + genre_data

    price = model.predict([data])[0]
    return {'price': price}
