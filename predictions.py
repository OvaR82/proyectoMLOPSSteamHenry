import json
import ast
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib


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


#desanidar genreros 
df_unnested = df.explode('genres')

dummy_df = pd.get_dummies(df_unnested['genres'], prefix='genres')

df_with_dummies = pd.concat([df_unnested, dummy_df], axis=1)

columns_to_drop = ['publisher', 'genres', 'url', 'release_date', 'tags', 'discount_price',
                   'reviews_url', 'specs', 'early_access', 'id', 'developer', 'sentiment']
#elimino columnas pra el modelo
df_reduced = df_with_dummies.drop(columns_to_drop, axis=1)

# dummy_app_name = pd.get_dummies(df_reduced['app_name'], prefix='app_name')
# df_reduced = pd.concat([df_reduced, dummy_app_name], axis= 1)
colums_to_drop_twice = ['app_name', 'title'] #elimino columnas que no sean dummies
df_dummies_for_model = df_reduced.drop(colums_to_drop_twice,axis=1)


#Defino features y target
X = df_dummies_for_model.drop('price', axis=1)
y = df_dummies_for_model['price']

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#modelo de machine learing
model = LinearRegression()

#entrenar el modelo
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# print("Mean Absolute Error:", mae)
# print("Mean Squared Error:", mse)
# print("R-squared:", r2)
# coefficients = pd.Series(model.coef_, index=X.columns)
# print(coefficients)

joblib.dump(model,'linear_regression_model.pkl')

feature_data_types = X_train.dtypes

# Print the data types of the features
print(feature_data_types)