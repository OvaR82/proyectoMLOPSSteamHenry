import json
import ast
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pickle


#lectura del json y creación df_unnested frame
rows = []
with open('steam_games.json') as f: 
    rows.extend(ast.literal_eval(line) for line in f)
df = pd.DataFrame(rows)


#Limpieza de df_unnested
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


#desanidar generos 
df_unnested = df.explode('genres')

# Convertir 'release_date' a año
df_unnested['release_year'] = pd.to_datetime(df_unnested['release_date']).dt.year

# Convertir 'genres' a números (usando one-hot encoding)
df_unnested = pd.get_dummies(df_unnested, columns=['genres'], prefix='', prefix_sep='')

# Split en entrenamiento y prueba
X = df_unnested[['release_year', 'metascore'] + list(df_unnested.columns[df_unnested.columns.str.contains('genres')])]
y = df_unnested['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Evaluar modelo
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Guardar modelo
pickle.dump(model, open('predictions.pkl', 'wb'))