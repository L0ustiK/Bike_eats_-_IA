import random
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine
import psycopg2

DATE_MIN = [2020,1,1]
DATE_MAX = [2023,4,4]

def add_date(date_min, date_max):
    
    (year,month,days) = (random.randint(date_min[0],date_max[0]),
                         random.randint(date_min[1],date_max[1]),
                         random.randint(date_min[2],date_max[2]))


    data_string = f"{year}-{month}-{days}"
    sql_date = datetime.strptime(data_string, "%Y-%m-%d").date()

    return sql_date

def add_random_object(names_list):

    name = names_list[random.randint(0, len(names_list)-1)]

    return name
     
def add_places(places_list):
     
     number = random.randint(1,45)
     place = places_list[random.randint(0, len(places_list)-1)]

     return str(number) + place + 'Paris'

def add_categories(categories_list):
    categorie = categories_list[random.randint(0, len(categories_list)-1)]

    return categorie

def open_file(file_name, list):
    file = open(file_name, "r", encoding="UTF8")
    for obj in file:
           list.append(obj[:-1])
    file.close()


def add_database(dataframe,table_name, db_connection):
    try:
        dataframe.to_sql(table_name, con = db_connection, if_exists='replace', chunksize=1000, method=None, index= False)
    except ValueError as vx:
        print(vx)
    except Exception as ex:
        print(ex)
    else:
        print("Add ", table_name, "table in data-base")

if __name__ == '__main__':

    tmp_name_lst = []; tmp_surname_lst = []; tmp_place_list = []; tmp_categories_list = []

    users_date = []; users_names = []; users_surnames = []
    restaurant_categories = []

    
    open_file('prenoms.txt', tmp_surname_lst)
    open_file('noms.txt', tmp_name_lst)
    open_file('categorie.txt', tmp_categories_list)


    for i in range(100):
        users_date.append(add_date(DATE_MIN, DATE_MAX))
        users_names.append(add_random_object(tmp_name_lst))
        users_surnames.append(add_random_object(tmp_surname_lst))    

    df_users = pd.DataFrame()
    df_users['nom'] = users_names
    df_users['prenom'] = users_surnames
    df_users['date_inscription'] = users_date
    df_users.insert(0, 'ID', range(1, 1 + len(df_users)))


    df_restaurant = pd.read_csv("restaurants-casvp.csv", sep = ';', encoding='utf-8')
    for i in range(len(df_restaurant.index)):
        restaurant_categories.append(add_categories(tmp_categories_list))

    df_restaurant['Principal_categorie'] = restaurant_categories
    df_restaurant.insert(0, 'ID', range(1, 1 + len(df_restaurant)))

    '''
    data_base = "db_bike_eat"
    host_name = "localhost"
    user_name = "leo"
    user_password = "leo123!"

    #Création de l'url de connection
    db_connection = create_engine("postgresql+psycopg2://" + user_name + ":" + user_password + "@" + host_name + "/" + data_base)

    #Connection pour executer les query de sauvegarde
    conn = db_connection.connect()


    add_database(df_restaurant, 'restaurant', db_connection)
    add_database(df_users, 'users', db_connection)

    
    conn.close()

    '''