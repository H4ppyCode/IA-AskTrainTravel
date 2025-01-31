import json
import random
import re
import os
from sklearn.model_selection import train_test_split

data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))
dataset_folder = os.path.join(data_folder, 'dataset')

# Load the list of cities
with open(os.path.join(data_folder, 'pathfinding', 'liste-des-gares.geojson'), 'r') as file:
    gares_data = json.load(file)

# Path to the CSV file
template_valid_sentence = os.path.join(dataset_folder, 'template_valid_sentence.csv')
template_invalid_sentence = os.path.join(dataset_folder, 'template_invalid_sentence.csv')
test_file = os.path.join(dataset_folder, 'test.json')
train_file = os.path.join(dataset_folder, 'train.json')

# Extract cities
communes = [feature['properties']['commune'] for feature in gares_data['features']]

# List of fictional city names
fictional_cities = [
    "Lumoria", "Arcanthia", "Nymeris", "Velindor", "Kalthera",
    "Zephyros", "Morivale", "Carathos", "Eldarion", "Froswyn",
    "Halcyra", "Draconis", "Aeloria", "Thalvarin", "Pyranos",
    "Novorion", "Myrelia", "Solenith", "Crystara", "Eryndor",
    "Ismeria", "Duskholm", "Valendreth", "Celestara", "Vorathis",
    "Xandralis", "Lysanthor", "Umbraketh", "Ferindor", "Quireth",
    "Tyveron", "Lunastra", "Obelith", "Arvendale", "Zorathia",
    "Kairalith", "Brynwald", "Ostaris", "Nerivale", "Florindel",
    "Harvion", "Serenthis", "Solwynne", "Graemora", "Kynthoria",
    "Lysendria", "Oranthal", "Velmara", "Trisendil", "Aerenthar"
]


print("Nombre de villes:", len(communes))
print("Exemples de villes:", communes[:5])

# load sentence templates from csv
templates_valids = []
with open(template_valid_sentence, 'r') as file:
    for line in file:
        templates_valids.append(line.strip())
templates_invalids = []
with open(template_invalid_sentence, 'r') as file:
    for line in file:
        templates_invalids.append(line.strip())


def get_random_city():
    all_cities = communes + fictional_cities
    ville_depart = random.choice(all_cities)
    ville_destination = random.choice(all_cities)
    while ville_depart == ville_destination:
        print(ville_depart, ville_destination)
        ville_destination = random.choice(all_cities)

    # Remove content within parentheses
    ville_depart = re.sub(r'\(.*?\)', '', ville_depart).strip()
    ville_destination = re.sub(r'\(.*?\)', '', ville_destination).strip()

    return ville_depart, ville_destination

# Generate dataset
def generate_random_valid_sentence():
    ville_depart, ville_destination = get_random_city()
    template = random.choice(templates_valids)
    sentence = template.replace("[ville_depart]", ville_depart).replace("[ville_destination]", ville_destination)
    # Find start and end positions of entities
    ville_depart_match = re.search(r"\b%s\b" % ville_depart, sentence)
    start_ville_depart = ville_depart_match.start()
    end_ville_depart = ville_depart_match.end()
    ville_destination_match = re.search(r"\b%s\b" % ville_destination, sentence)
    start_ville_destination = ville_destination_match.start()
    end_ville_destination = ville_destination_match.end()
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

# Generate x valid sentences
valid_sentences = [generate_random_valid_sentence() for _ in range(5000)]

# Generate x invalid sentences
invalid_sentences = [generate_random_invalid_sentence() for _ in range(5000)]

# Combine and shuffle the sentences
sentences = valid_sentences + invalid_sentences
random.shuffle(sentences)

# Split dataset into training and testing sets
train_sentences, test_sentences = train_test_split(sentences, test_size=0.2, random_state=42)

# Save training sentences to a JSON file
with open(train_file, 'w') as file:
    json.dump(train_sentences, file, ensure_ascii=False, indent=4)

# Save testing sentences to a JSON file
with open(test_file, 'w') as file:
    json.dump(test_sentences, file, ensure_ascii=False, indent=4)

print("Training sentences saved to %s" % train_file)
print("Testing sentences saved to %s" % test_file)
