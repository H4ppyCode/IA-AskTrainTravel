import csv
import json
import random

# Path to the CSV file
input_file_valid_sentence = 'validation_valid_sentence.csv'
input_file_invalid_sentence = 'validation_invalid_sentence.csv'
output_file = 'validation.json'

# Initialize the list of sentences in NER format
ner_data = []

with open(input_file_valid_sentence, mode='r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        sentence = row['sentence'].strip()
        depart = row['from'].strip()
        destination = row['to'].strip()

        start_depart = sentence.index(depart)
        end_depart = start_depart + len(depart)
        start_destination = sentence.index(destination)
        end_destination = start_destination + len(destination)

        # Build the entities
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

with open(input_file_invalid_sentence, mode='r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=',')
    for row in reader:
        sentence = row['sentence'].strip()

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