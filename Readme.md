# T-AIA (AYAAAAAAAAAAAAAA)

## For documentation & Work Progress

[![Notion](https://img.shields.io/badge/Notion-000000?style=for-the-badge&logo=notion&logoColor=white)
](https://alpine-cuckoo-e2f.notion.site/T-AIA-901-c6ed78595e584f15847a8320bc7a1f8c)


## Project 

The "Travel Order Resolver" project aims to develop a program that processes travel orders in natural language,
identifying departure and destination locations (French trains) in text (or optionally voice) and determining the best train itinerary.
The main focus is on Natural Language Processing (NLP) to extract relevant information from French text and optimize travel routes. 
Additionally, the project requires building a custom training dataset for the NLP model and evaluating its performance.

The key steps include:

    1. A voice recognition step to convert audio data into text.
    2. Natural Language Processing (NLP) to understand the orders.
    3. An optimization process to find the best train connections.




## Stack using

![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)
![SpaCy](https://img.shields.io/badge/Spacy-spacy?style=for-the-badge&logo=spacy&logoColor=black&logoSize=auto&color=blue)

## Initialization

### Python environnment

Setup a python venv and install the required packages:

_Inside the venv_

```sh
python -m pip install -r requirements.txt
python -m spacy download fr_core_news_md
```

### SNCF Datasets

1. GTFS for pathfinding

Download the GTFS SNCF dataset for the [TER](https://www.data.gouv.fr/fr/datasets/horaires-des-lignes-ter-sncf/) and [TGV](https://www.data.gouv.fr/fr/datasets/horaires-des-tgv/) lines and merge them (just append the TGV file at the end of the TER files, without the header of the TGV files)
The required files are:

- routes.txt
- stop_times.txt
- stops.txt
- trips.txt

2. Json for railway tracks path (display only)

Download the [railway tracks speed (json)](https://data.sncf.com/explore/dataset/vitesse-maximale-nominale-sur-ligne/export/) and the [list of stations (geojson)](https://transport.data.gouv.fr/datasets/liste-des-gares) and save them in the same directory as the GTFS data.

## To Use

`python nlp_processor.py sample_nlp_input.txt'` for processing the input file with spaCy NLP
