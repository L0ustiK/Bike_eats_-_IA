import nltk
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess(ingredients_list):
    text = ','.join(ingredients_list)
    # Convertir en minuscules
    text = text.lower()
    # Supprimer la ponctuation
    text = ''.join([c for c in text if c not in ('!', '.', ':', ',', ';', '?', '-', '_', '/', '\\', '(', ')', '[', ']', '{', '}')])
    # Supprimer les mots vides
    text = ' '.join([word for word in text.split() if word not in stop_words])
    # Lemmatiser
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text


def encoding_one_hot(file_name):

    df =  pd.read_csv(file_name, sep=",", encoding="utf-8")

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

def  naive_bayes_models(file_name):

    data =  pd.read_json(file_name)

    #plat_columns = ['Plat '+str(n) for n in range(1,len(data.columns)-1)]

    #one_tab = pd.melt(data, id_vars=['Categorie'], value_vars=plat_columns, var_name='plat col', value_name='plat')

    #one_tab = one_tab.dropna(subset=['plat'])
    
    x = data['ingredients']
    y = data['cuisine']
    train_data,test_data,train_labels,test_labels = train_test_split(x,y, test_size=0.2, random_state=42)

    train_data = train_data.apply(preprocess)
    test_data = test_data.apply(preprocess)


    vectorizer = CountVectorizer()
    x_train_vectorized = vectorizer.fit_transform(train_data)
    x_test_vectorized = vectorizer.transform(test_data)
    

    model = MultinomialNB()
    model.fit(x_train_vectorized, train_labels)

    y_pred = model.predict(x_test_vectorized)

    scores = cross_val_score(model, x_train_vectorized, train_labels, cv=10)
    print("Scores de validation croisée :", scores)
    print("Moyenne des scores de validation croisée :", scores.mean())
    print('\n')


def decision_tree_classifier(file_name):


    data = pd.read_json(file_name)

    #plat_columns = ['Plat '+str(n) for n in range(1,len(data.columns)-1)]

    #one_tab = pd.melt(data, id_vars=['Categorie'], value_vars=plat_columns, var_name='plat col', value_name='plat')
    #one_tab = one_tab.dropna(subset=['plat'])

    x = data['ingredients']
    y = data['cuisine']

    train_data, test_data, train_labels, test_labels = train_test_split(x, y, test_size=0.2, random_state=42)

    train_data = train_data.apply(preprocess)
    test_data = test_data.apply(preprocess)


    vectorizer = CountVectorizer()
    x_train_vectorized = vectorizer.fit_transform(train_data)
    x_test_vectorized = vectorizer.transform(test_data)

    model = DecisionTreeClassifier(max_depth=250, min_samples_split=2)
    model.fit(x_train_vectorized, train_labels)

    y_pred = model.predict(x_test_vectorized)

    scores = cross_val_score(model, x_train_vectorized, train_labels, cv=3)
    print("Scores de validation croisée :", scores)
    print("Moyenne des scores de validation croisée :", scores.mean())

    print('\n')



def predict_dish_category_with_ingredients(dish_name, ingredients, model, vectorizer):
    dish_name_and_ingredients = dish_name + " " + " ".join(ingredients)
    preprocessed_dish = preprocess(dish_name_and_ingredients)
    dish_vectorized = vectorizer.transform([preprocessed_dish])
    predicted_category = model.predict(dish_vectorized)
    return predicted_category[0]



def logistic_regression_classifier(file_name):
    data = pd.read_json(file_name)

    x = data['ingredients']
    y = data['cuisine']

    train_data, test_data, train_labels, test_labels = train_test_split(x, y, test_size=0.25, random_state=3)

    train_data = train_data.apply(preprocess)
    test_data = test_data.apply(preprocess)

    vectorizer = CountVectorizer()
    x_train_vectorized = vectorizer.fit_transform(train_data)
    x_test_vectorized = vectorizer.transform(test_data)

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train_vectorized, train_labels)

    y_pred = model.predict(x_test_vectorized)

    scores = cross_val_score(model, x_train_vectorized, train_labels, cv=3)
    print("Scores de validation croisée :", scores)
    print("Moyenne des scores de validation croisée :", scores.mean())
    print('\n')
