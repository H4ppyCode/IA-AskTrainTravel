import sys
import spacy
import re

# Charger le modèle de langue française
nlp = spacy.load("fr_core_news_md")


def is_french_trip(sentence):
    doc = nlp(sentence)
    if doc.lang_ != "fr":
        return 'NOT_FRENCH'

    # Check if there are at least two locations in the sentence
    locations = [ent for ent in doc.ents if ent.label_ == "LOC"]
    if len(locations) >= 2:
        return 'TRIP'
    return 'NOT_TRIP'


def extract_trip_info(sentence):
    doc = nlp(sentence)
    locations = [ent.text for ent in doc.ents if ent.label_ == "LOC"]

    if len(locations) >= 2:
        departure = locations[0]
        arrival = locations[1]
        return departure, arrival
    return None, None


def process_sentences(input_file):
    results = ['sequence_id,departure,arrival']
    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            sentence_id, sentence = line.strip().split(',', 1)
            trip_status = is_french_trip(sentence)
            if trip_status == 'TRIP':
                departure, arrival = extract_trip_info(sentence)
                results.append(f"{sentence_id},{departure},{arrival}")
            else:
                results.append(f"{sentence_id},{trip_status},")
    return results


if __name__ == "__main__":
    # if len(sys.argv) != 1:
        # print("Usage: python nlp_processor.py <input_file>")
        # sys.exit(1)

    input_file = sys.stdin.read()
    departure, arrival = extract_trip_info(input_file)
    # print(f"{departure} {arrival}")
    print(departure)
    print(arrival)
    # results = process_sentences(input_file)
    # for result in results:
    #     print(result)
