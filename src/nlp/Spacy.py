from typing import Union, Optional, Callable, List
import argparse
import glob
import logging
import sys
import spacy
from enum import Enum
from dataclasses import dataclass
from nlp.IsTripModel import SpacyIsTripModel
from nlp.GetTripModel import SpacyGetTripModel

class SpacyNamespace(argparse.Namespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model: str
        self.is_trip_model: str
        self.input_files: str = None
        self.use_stt: bool = False
        self.use_stdin: bool = False
        self.no_pathfinding: bool = False
        self.loaded_is_trip_model: SpacyIsTripModel = None
        self.loaded_model: SpacyGetTripModel = None

    def validate(self, is_runned_as_subcommand: bool) -> bool:
        if not self.input_files and not self.use_stt and not self.use_stdin:
            if not is_runned_as_subcommand:
                logging.info("No nlp input provided, fallback to Speech To Text module.")
                self.use_stt = True
            else:
                logging.info("No nlp input provided, fallback to stdin.")
                self.use_stdin = True
        # Should not happen because of the mutual exclusion group
        elif int(self.use_stt) + int(self.use_stdin) + int(bool(self.input_files)) > 1:
            logging.error("Multiple nlp inputs provided, please choose only one.")
            return False
        try:
            self.loaded_is_trip_model = SpacyIsTripModel(self.is_trip_model)
        except Exception as e:
            logging.error(f"Error loading is_trip model {self.is_trip_model}: {e}")
            return False
        try:
            self.loaded_model = SpacyGetTripModel(self.model)
        except Exception as e:
            logging.error(f"Error loading model {self.model}: {e}")
            return False
        
        return True


def fill_spacy_parser(parser: argparse.ArgumentParser, args_prefix: str = "") -> None:
    parser.add_argument(f"--{args_prefix}model", default="fr_core_news_md", help="Spacy model name to use. Default: fr_core_news_md", type=str)
    parser.add_argument(f"--{args_prefix}is-trip-model", default="fr_core_news_md", help="Spacy model name to use. Default: fr_core_news_md", type=str)
    
    nlp_input_group = parser.add_mutually_exclusive_group(required=False)
    nlp_input_group.add_argument(f"--{args_prefix}input-files", default=None, help="Glob pattern for sentence text files.", type=str)
    # Can't use stt if we use only the NLP feature
    if args_prefix:
        nlp_input_group.add_argument(f"--{args_prefix}use-stt", default=False, action='store_true', help="Use Speech-To-Text output as NLP input.")
        parser.add_argument(f"--{args_prefix}no-pathfinding", default=False, action='store_true', help="Do not use pathfinding module.")
        
    nlp_input_group.add_argument(f"--{args_prefix}use-stdin", default=False, action='store_true', help="Use stdin as input.")


class TripStatus(Enum):
    NOT_FRENCH = "NOT_FRENCH"
    NOT_TRIP = "NOT_TRIP"
    TRIP = "TRIP"

    def __str__(self):
        return self.value

@dataclass
class Trip:
    departure: str
    arrival: str
    id: Optional[int] = None

@dataclass
class TripResponse:
    sentence: str
    status: TripStatus
    trip: Optional[Trip]
    id: Optional[int]

    @property
    def is_trip(self) -> bool:
        return self.status == TripStatus.TRIP
    
    def to_csv_line(self, id: Optional[int] = None, with_source: bool = False) -> str:
        # If the trip has an id, use it, otherwise use the provided id
        if self.id:
            id = self.id
        prefix = f"{id}," if id is not None else ""
        if self.is_trip:
            result = f"{prefix}{self.trip.departure},{self.trip.arrival}"
        else:
            result = f"{prefix}{self.status},"
        if with_source:
            result += f",{self.sentence}"
        return result

NlpSentence = Union[str, spacy.language.Doc]


class Spacy:
    def __init__(self, config: SpacyNamespace):
        self.config = config

    @property
    def model(self) -> SpacyGetTripModel:
        return self.config.loaded_model
    
    @property
    def is_trip_model(self) -> SpacyIsTripModel:
        return self.config.loaded_is_trip_model
    
    def is_french(self, sentence: NlpSentence) -> bool:
        return self.model.normalize_sentence(sentence).lang_ == "fr"
    
    def is_trip(self, sentence: NlpSentence) -> bool:
        """Sentence must be either str or result of is_trip_model(sentence)"""
        return self.is_trip_model.is_true_sentence(sentence)

    def get_trip_status(self, sentence: NlpSentence) -> TripStatus:
        # Check if the sentence is a trip request
        evaluated_sentence = self.is_trip_model.normalize_sentence(sentence)
        if not self.is_trip(evaluated_sentence):
            return TripStatus.NOT_TRIP
    
        # Check if the sentence is in french
        if not self.is_french(evaluated_sentence):
            return TripStatus.NOT_FRENCH

        return TripStatus.TRIP
    
    def process_sentence(self, sentence: NlpSentence, id: Optional[int] = None) -> TripResponse:
        sentence_lower = str(sentence).lower()
        response = TripResponse(str(sentence), TripStatus.NOT_TRIP, None, id)

        # Use the is_trip model to check if the sentence is a valid trip request
        response.status = self.get_trip_status(sentence_lower)
        if not response.is_trip:
            return response
        
        # Can safely assume that there are two locations
        evaluated = self.model.normalize_sentence(sentence_lower)
        response.trip = Trip(self.model.get_departure_city(evaluated), self.model.get_destination_city(evaluated))
        return response
    
    def process_sentences_file(self, input_file: str) -> List[TripResponse]:
        results: List[TripResponse] = []
        with open(input_file, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                if not "," in line:
                    logging.warning(f"{input_file}:{i} - No sequence id, skipping line: '{line}'", file=sys.stderr)
                    continue
                sentence_id, sentence = line.strip().split(',', 1)
                try:
                    sentence_id = int(sentence_id)
                except ValueError:
                    logging.warning(f"{input_file}:{i} - Invalid sequence id, skipping line: '{line}'", file=sys.stderr)
                    continue
                results.append(self.process_sentence(sentence, int(sentence_id)))
        return results
    
    def trips_to_csv(self, trips: List[TripResponse], add_source: bool = False) -> str:
        result = ['sequence_id,departure,arrival']
        for i, trip in enumerate(trips):
            result.append(trip.to_csv_line(i, with_source=add_source))
        return '\n'.join(result)
    
    def run(self, on_trip: Callable[[TripResponse], None] = None):
        if self.config.use_stt:
            raise Exception("Spacy module can't be used with Speech To Text module (must be managed in AYA.py).")
        if self.config.use_stdin:
            while True:
                try:
                    sentence = input("Enter a sentence: ")
                except (EOFError, KeyboardInterrupt):
                    logging.error("No sentence provided.")
                    return 1
                trip = self.process_sentence(sentence)
                if trip.is_trip:
                    print("Trip: departure=%s, arrival=%s" % (trip.trip.departure, trip.trip.arrival))
                else:
                    print("Not a valid trip: %s" % trip.status)
                if on_trip:
                    try:
                        on_trip(trip)
                    except Exception as e:
                        logging.error("Error processing trip: %s", e)
        else:
            for file in glob.glob(self.config.input_files):
                logging.info("Processing file: ", file)
                trips = self.process_sentences_file(file)
                print(self.trips_to_csv(trips, add_source=True))
                if on_trip:
                    for trip in trips:
                        try:
                            on_trip(trip)
                        except Exception as e:
                            logging.error("Error processing trip: %s", e)
            logging.info("No more files to process.")
        return 0



def main():
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] - %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
                    prog='SNCF-Pathfinding',
                    description='Compute the shortest path between two cities in France')
    fill_spacy_parser(parser)
    args: SpacyNamespace = parser.parse_args()
    pth = Spacy(args)
    pth.run()

if __name__ == '__main__':
    main()
