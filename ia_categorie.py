import pandas as pd
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense


def encoding_one_hot(file_name):

    df = pd.read_csv(file_name, sep=',', encoding='utf8')
    
    # Récup catégorie
    categories = df.iloc[:,1:]
    
    # Garder 1 seul exemplaire
    unique_categories = categories.stack().unique()

    # Créer un dictionnaire qui associe chaque catégorie à un index
    cat_to_index = {cat: i for i, cat in enumerate(unique_categories)}

    # Encoder les données en vecteurs one-hot
    one_hot = np.zeros((len(categories), len(unique_categories)))
    for i, row in categories.iterrows():
        for cat in row:
            if cat in cat_to_index:
                one_hot[i, cat_to_index[cat]] = 1

    # Ajouter les vecteurs one-hot au DataFrame d'origine
    one_hot_df = pd.DataFrame(one_hot, columns=unique_categories)

    one_hot_df = one_hot_df.astype(int)
    result = pd.concat([df.iloc[:, 0], one_hot_df], axis=1)

    return result

def neuron_network(file_name):

    data = pd.read_csv(file_name, sep=',', encoding='unicode_escape')

    data.dtypes
    
    x = data.drop('Categorie', axis = 1)
    y = data['Categorie']

    train_data, test_data, train_labels, test_labels = train_test_split(x,y, test_size=0.2, random_state=20)

    model = Sequential()
    model.add(Dense(64, activation="relu", input_shape=(len(train_data.columns),)))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    # Compiler le modèle
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Entraîner le modèle
    model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_data=(test_data, test_labels))




