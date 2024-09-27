# T-AIA (AYAAAAAAAAAAAAAA)

## For documentation & Work Progress

[Notion](https://alpine-cuckoo-e2f.notion.site/T-AIA-901-c6ed78595e584f15847a8320bc7a1f8c)

## Stack using

![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)
![SpaCy](https://img.shields.io/badge/Spacy-spacy?style=for-the-badge&logo=spacy&logoColor=black&logoSize=auto&color=blue)

## Initialization

### Python environnment

Setup a python venv and install the required packages:

_Inside the venv_

```sh
python -m pip install -r requirements.txt
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
