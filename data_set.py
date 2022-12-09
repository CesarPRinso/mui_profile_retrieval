import json
from datetime import datetime
import random


def read_json_kaggle():
    with open('kaggle_dataset.json') as json_file:
        data = json.load(json_file)

    return data['articles']


def get_data():
    categories = []
    dates = []

    categorias_unicas = list(set(categories))
    dates_list = list(set(dates))

    data = read_json_kaggle()

    for dct in data:
        categories.append(dct['category'])

    for dct in data:
        if dct['category'] in ['POLITICS', 'ENTERTAINMENT']:
            dates.append(datetime.strptime(dct['date'], "%Y-%m-%d").strftime("%d-%m-%Y"))

    print("-------------------")
    print("- Análisis  en el conjunto de datos -")
    print("-------------------")
    print(f'tenemos  {len(categorias_unicas)} categorias')
    print(categorias_unicas)
    print("-------------------")
    print('la fecha más actualizada del data set es : ' + str(max(dates)))
    print("-------------------")
    print(
        "- Tómaremos  las categorías -- > TECH, ENTERTAINMENT, BUSINESS, SPORTS, SCIENCE, POLITICS- y datos más actuales, la fecha mayores de 01-01-2017")
    print("-------------------")


    filtered_data = []
    filter_date = datetime.strptime('2017-01-01', "%Y-%m-%d")
    allowed_categories = {'TECH': 0, 'ENTERTAINMENT': 0, 'BUSINESS': 0, 'SPORTS': 0, 'SCIENCE': 0, 'POLITICS': 0}
    for dct in data:
        if dct['category'] in allowed_categories:
            datetime_object = datetime.strptime(dct['date'], '%Y-%m-%d')
            if dct['category'] in ['POLITICS', 'ENTERTAINMENT']:
                if datetime_object >= filter_date:
                    filtered_data.append(dct)
                    allowed_categories[dct['category']] += 1
            else:
                filtered_data.append(dct)
                allowed_categories[dct['category']] += 1

    print(allowed_categories)
    print("--")

    politics = []
    entertainment = []
    tech = []
    business = []
    sports = []
    science = []
    for dct in filtered_data:
        if dct['category'] == "POLITICS":
            politics.append(tuple((str(dct['headline']) + '. ' + str(dct['short_description']), "POLITICS")))
        if dct['category'] == "ENTERTAINMENT":
            entertainment.append(tuple((str(dct['headline']) + '. ' + str(dct['short_description']), "ENTERTAINMENT")))
        if dct['category'] == "TECH":
            tech.append(tuple((str(dct['headline']) + '. ' + str(dct['short_description']), "TECH")))
        if dct['category'] == "BUSINESS":
            business.append(tuple((str(dct['headline']) + '. ' + str(dct['short_description']), "BUSINESS")))
        if dct['category'] == "SPORTS":
            sports.append(tuple((str(dct['headline']) + '. ' + str(dct['short_description']), "SPORTS")))
        if dct['category'] == "SCIENCE":
            science.append(tuple((str(dct['headline']) + '. ' + str(dct['short_description']), "SCIENCE")))

    max_samples = 2000
    politics = random.sample(politics, max_samples)
    entertainment = random.sample(entertainment, max_samples)
    business = random.sample(business, max_samples)
    sports = random.sample(sports, max_samples)
    science = random.sample(science, max_samples)
    tech = random.sample(tech, max_samples)
    articles = politics + entertainment + business + sports + science + tech
    print(
        count_samples_per_cat(articles, ['TECH', 'ENTERTAINMENT', 'BUSINESS', 'SPORTS', 'SCIENCE', 'POLITICS']))
    return articles


def count_samples_per_cat(samples, cats):
    counts = {}
    for name in cats:
        counts[name] = 0

    for corpus, label in samples:
        counts[label] += 1

    return counts
