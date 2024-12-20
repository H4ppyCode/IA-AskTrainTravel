import csv
import json
import random

# Path to the CSV file
input_file_kaggle = 'dataframe.csv'
input_file_bad_sentence = 'phrases_dataset.csv'
output_file = 'validation.json'

# Initialize the list of sentences in NER format
ner_data = []

# Read the CSV file
with open(input_file_kaggle, mode='r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=';')
    for row in reader:
        sentence = row['text'].strip()
        depart = row['from'].strip()
        destination = row['to'].strip()

        # Identify the positions of the entities
        start_depart = sentence.index(depart)
        end_depart = start_depart + len(depart)
        start_destination = sentence.index(destination)
        end_destination = start_destination + len(destination)

        # Build the dictionary for each sentence
        entities = [
            {
                "start": start_depart,
                "end": end_depart,
                "label": "VILLE_DEPART"
            },
            {
                "start": start_destination,
                "end": end_destination,
                "label": "VILLE_DESTINATION"
            }
        ]

        ner_data.append({
            "sentence": sentence,
            "entities": entities
        })

with open(input_file_bad_sentence, mode='r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=',')
    for row in reader:
        sentence = row['Phrase'].strip()

        ner_data.append({
            "sentence": sentence,
            "entities": []
        })

# Shuffle the sentences
random.shuffle(ner_data)

# Write the results in JSON format
with open(output_file, mode='w', encoding='utf-8') as jsonfile:
    json.dump(ner_data, jsonfile, ensure_ascii=False, indent=4)

print(f"NER data has been written to {output_file}.")