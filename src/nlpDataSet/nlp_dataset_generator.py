import json
import random
from sklearn.model_selection import train_test_split

# Charger les données de villes
with open('../pathfinding/gtfs/liste-des-gares.geojson', 'r') as file:
    gares_data = json.load(file)

# Extract cities
communes = [feature['properties']['commune'] for feature in gares_data['features']]

print("Nombre de villes:", len(communes))
print("Exemples de villes:", communes[:5])

# sentence templates
templates = [
    "Je veux aller de [ville_depart] à [ville_destination].",
    "Je pars de [ville_depart] pour aller à [ville_destination].",
    "Je vais de [ville_depart] à [ville_destination].",
    "Je prends le train de [ville_depart] à [ville_destination].",
    "Le train part de [ville_depart] pour [ville_destination].",
    "Je prends un train depuis [ville_depart] vers [ville_destination].",
    "Je voyage de [ville_depart] à [ville_destination].",
    "Voyage de [ville_depart] à [ville_destination].",
    "Je pars en voyage de [ville_depart] pour [ville_destination].",
    "Le départ est de [ville_depart], l'arrivée est à [ville_destination].",
    "Le trajet commence à [ville_depart] et se termine à [ville_destination].",
    "Départ de [ville_depart], arrivée à [ville_destination].",
    "Je dois partir de [ville_depart] pour aller à [ville_destination].",
    "Nous partons de [ville_depart] en direction de [ville_destination] me détendre.",
    "Je quitte [ville_depart] pour rejoindre [ville_destination].",
    "Je me dirige de [ville_depart] vers [ville_destination].",
    "Direction [ville_destination] en partant de [ville_depart].",
    "En route de [ville_depart] pour [ville_destination].",
    "Depuis [ville_depart], je vais vers [ville_destination].",
    "De [ville_depart] vers [ville_destination], je prends le train.",
    "Partir de [ville_depart] pour aller vers [ville_destination].",
    "Mon trajet commence à [ville_depart] et finit à [ville_destination].",
    "Trajet en train de [ville_depart] à [ville_destination].",
    "Je fais le trajet de [ville_depart] jusqu'à [ville_destination] pour prendre un monaco."
]

# Generate dataset
def generate_random_sentence():
    ville_depart = random.choice(communes)
    ville_destination = random.choice(communes)
    template = random.choice(templates)
    sentence = template.replace("[ville_depart]", ville_depart).replace("[ville_destination]", ville_destination)

    # Find start and end positions of entities
    start_ville_depart = sentence.find(ville_depart)
    end_ville_depart = start_ville_depart + len(ville_depart)
    start_ville_destination = sentence.find(ville_destination)
    end_ville_destination = start_ville_destination + len(ville_destination)
    sentence = sentence.lower()
    return {
        "sentence": sentence,
        "entities": [
            {"start": start_ville_depart, "end": end_ville_depart, "label": "VILLE_DEPART"},
            {"start": start_ville_destination, "end": end_ville_destination, "label": "VILLE_DESTINATION"}
        ]
    }

# Generate and store 1000 random sentences
sentences = [generate_random_sentence() for _ in range(1000)]

# Split dataset into training and testing sets
train_sentences, test_sentences = train_test_split(sentences, test_size=0.2, random_state=42)

# Save training sentences to a JSON file
with open('train.json', 'w') as file:
    json.dump(train_sentences, file, ensure_ascii=False, indent=4)

# Save testing sentences to a JSON file
with open('test.json', 'w') as file:
    json.dump(test_sentences, file, ensure_ascii=False, indent=4)

print("Training sentences saved to train.json")
print("Testing sentences saved to test.json")




