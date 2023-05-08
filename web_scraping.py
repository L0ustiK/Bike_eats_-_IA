import requests
import json
import pandas as pd

API_KEY = ""

country_lst = [ 'American'
                ,'British'
                ,'Cajun'
                ,'Caribbean'
                ,'Chinese'
                ,'Eastern European'
                ,'European'
                ,'French'
                ,'German'
                ,'Greek'
                ,'Indian'
                ,'Irish'
                ,'Italian'
                ,'Japanese'
                ,'Jewish'
                ,'Korean'
                ,'Latin American'
                ,'Mediterranean'
                ,'Mexican'
                ,'Middle Eastern'
                ,'Nordic'
                ,'Southern'
                ,'Spanish'
                ,'Thai'
                ,'Vietnamese']

def get_ingredients(recipe_id, api_key):
    url = f'https://api.spoonacular.com/recipes/{recipe_id}/ingredientWidget.json'
    query_params = {'apiKey': api_key}
    response = requests.get(url, params=query_params)
    ingredients = []

    if response.status_code == 200:
        data = json.loads(response.text)
        for ingredient in data['ingredients']:
            ingredients.append(ingredient['name'])
    else:
        print(f"La requête d'ingrédients a échoué avec le code de statut : {response.status_code}")

    return ', '.join(ingredients)

def add_dish(country_zone, result_colmun, country_colmun, ingredients_colmun):
    url = f'https://api.spoonacular.com/recipes/complexSearch'

    query_params = {    
        'apiKey': API_KEY,
        'cuisine': country_zone,
        'number': 100, 
    }

    response = requests.get(url, params=query_params)

    if response.status_code == 200:
        data = json.loads(response.text)
        for recipe in data['results']:
            result_colmun.append(recipe['title'])
            country_colmun.append(country_zone)
            ingredients_colmun.append(get_ingredients(recipe['id'], API_KEY))

    else:
        print(f"La requête a échoué avec le code de statut : {response.status_code}")

if __name__ == '__main__':
    dish_col = [] ; country_col = [] ; ingredients_col = []

    for country in country_lst:
        add_dish(country, dish_col, country_col, ingredients_col)

    df = pd.DataFrame();

    df['country'] = country_col; df['dish'] = dish_col; df['ingredients'] = ingredients_col
    df.to_csv('data_set/Categorie.csv', sep=',', encoding='utf-8')

