import requests
import json
import pandas as pd

country_lst = ['African'
            ,'American'
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

def add_dish(country_zone, result_colmun, country_colmun):
    api_key = 'cc958560cb3a4b11bcdb9c9305de578f'
    url = f'https://api.spoonacular.com/recipes/complexSearch'

    query_params = {    
        'apiKey': api_key,
        'cuisine': country_zone,
        'number': 100, 
    }

    response = requests.get(url, params=query_params)

    if response.status_code == 200:
        data = json.loads(response.text)
        for recipe in data['results']:
            result_colmun.append(recipe['title'])
            country_colmun.append(country_zone)

    else:
        print(f"La requête a échoué avec le code de statut : {response.status_code}")

if __name__ == '__main__':
    dish_col = []
    country_col = []


    for country in country_lst:
        add_dish(country, dish_col, country_col)


    df = pd.DataFrame();

    df['country'] = country_col; df['dish'] = dish_col
    df.to_csv('Categorie_2.csv', sep=',', encoding='utf-8')
