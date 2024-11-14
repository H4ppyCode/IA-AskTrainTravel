import json
import random
from sklearn.model_selection import train_test_split

# Load the list of cities
with open('../pathfinding/gtfs/liste-des-gares.geojson', 'r') as file:
    gares_data = json.load(file)

# Extract cities
communes = [feature['properties']['commune'] for feature in gares_data['features']]

print("Nombre de villes:", len(communes))
print("Exemples de villes:", communes[:5])

# sentence templates
templates_valids = [
    "Je veux aller de [ville_depart] à [ville_destination] demain matin, avant 9h.",
    "Je pars de [ville_depart] pour aller à [ville_destination] ce week-end avec ma famille.",
    "Je vais de [ville_depart] à [ville_destination] pour un voyage d'affaires lundi prochain.",
    "Je prends le train de [ville_depart] à [ville_destination] le 14 avril à 14h30.",
    "Le train part de [ville_depart] pour [ville_destination] après la réunion à 18h.",
    "Je prends un train depuis [ville_depart] vers [ville_destination], espérant arriver avant minuit.",
    "Je voyage de [ville_depart] à [ville_destination], mais je vais m'arrêter en chemin.",
    "Voyage en première classe de [ville_depart] à [ville_destination] demain.",
    "Je pars en voyage de [ville_depart] pour [ville_destination] pendant mes vacances.",
    "Le départ est prévu à [ville_depart] à midi, et l'arrivée à [ville_destination] vers 16h.",
    "Le trajet commence à [ville_depart] et se termine à [ville_destination] ce vendredi soir.",
    "Départ de [ville_depart] à 7h, arrivée à [ville_destination] avant le coucher du soleil.",
    "Je dois partir de [ville_depart] pour aller à [ville_destination] pour un entretien à 10h.",
    "Nous partons de [ville_depart] en direction de [ville_destination] pour les vacances d'été.",
    "Je quitte [ville_depart] pour rejoindre [ville_destination] après la pluie.",
    "Je me dirige de [ville_depart] vers [ville_destination] en passant par une gare intermédiaire.",
    "Direction [ville_destination] en partant de [ville_depart] tôt le matin.",
    "En route de [ville_depart] pour [ville_destination], mais je vais faire une escale.",
    "Depuis [ville_depart], je vais vers [ville_destination] pour assister à une conférence.",
    "De [ville_depart] vers [ville_destination], je prends le train le plus rapide disponible.",
    "Partir de [ville_depart] pour aller vers [ville_destination] avec mes amis ce soir.",
    "Mon trajet commence à [ville_depart] et finit à [ville_destination] après plusieurs heures de route.",
    "Trajet en train de [ville_depart] à [ville_destination], avec un arrêt à mi-chemin.",
    "Je fais le trajet de [ville_depart] jusqu'à [ville_destination] pour prendre un monaco."
]

templates_invalids = [
    "Je veux aller de [ville_depart] à X pour voir des amis.",
    "Je vais de [ville_depart] au bout du monde après le travail.",
    "Je pars pour aller à [ville_destination], sans idée précise de l'heure.",
    "Voyage depuis [ville_depart] pour aller ailleurs sans vraiment savoir où.",
    "[ville_depart] et [ville_destination], c'est loin à pied ?",
    "Je veux aller de [ville_depart] sans savoir où je vais atterrir.",
    "Partir de [ville_depart] pour une aventure vers un lieu inconnu.",
    "[ville_destination], c'est où ça exactement ? Je n'ai jamais entendu parler."
]

def get_random_city():
    ville_depart = random.choice(communes)
    ville_destination = random.choice(communes)
    while ville_depart == ville_destination:
        ville_destination = random.choice(communes)
    return ville_depart, ville_destination

# Generate dataset
def generate_random_valid_sentence():
    ville_depart, ville_destination = get_random_city()
    template = random.choice(templates_valids)
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

def generate_random_invalid_sentence():
    ville_depart, ville_destination = get_random_city()
    template = random.choice(templates_invalids)
    sentence = template.replace("[ville_depart]", ville_depart).replace("[ville_destination]", ville_destination)

    return {
        "sentence": sentence.lower(),
        "entities": []
    }

# Generate 1000 valid sentences
valid_sentences = [generate_random_valid_sentence() for _ in range(1000)]

# Generate 1000 invalid sentences
invalid_sentences = [generate_random_invalid_sentence() for _ in range(1000)]

# Combine and shuffle the sentences
sentences = valid_sentences + invalid_sentences
random.shuffle(sentences)

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




