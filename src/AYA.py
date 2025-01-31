from typing import TYPE_CHECKING, Callable, Optional
import sys
import argparse
import glob
import logging

from stt.Whisper import Whisper, WhisperNamespace, fill_whisper_parser
from nlp.Spacy import Spacy, SpacyNamespace, fill_spacy_parser
from pathfinding.main import Pathfinder, PathfindingNamespace, fill_path_parser

if TYPE_CHECKING:
    from nlp.Spacy import TripResponse

def add_subparser(parser: argparse.ArgumentParser, 
                  subparsers: argparse._SubParsersAction, 
                  command: str,
                  title: str,
                  description: str,
                  fill_fct: Callable[[argparse.ArgumentParser, str], None]) -> None:
    # Create subparser
    subparser = subparsers.add_parser(command, help='Use only the %s feature' % command.upper())
    fill_fct(subparser, "")
    # Create subparser group
    parser = parser.add_argument_group(title=title, description=description)
    fill_fct(parser, "%s." % command)


def get_arg_parser():
    parser = argparse.ArgumentParser(
                    prog='AYA',
                    description="""Personal train trips planner from natual language.
Understand audio and text requests but only in French. May not do everything you ask but will try its best.""", epilog="AYA - All You Ask")
    subparsers = parser.add_subparsers(dest='subcommand', help='Sub command helps')
    

    add_subparser(parser, subparsers, 'stt', 'Speech-To-Text', 'Speech-To-Text parameters. Use this to provide audio input.', fill_whisper_parser)
    add_subparser(parser, subparsers, 'nlp', 'Natural Language Processing', 'Natural Language Processing parameters. Used to convert sentence in trip request. The nlp can read from stdin, text files, or whisper output. The parameters are mutually exclusive and default to whisper output if none is provided.', fill_spacy_parser)
    add_subparser(parser, subparsers, 'path', 'Pathfinding', 'Pathfinding parameters. Used to find the best trip between two cities. Will use the GTFS data of french TER and TGV trains.', fill_path_parser)
    

    return parser

class AYAParameters(argparse.Namespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stt = WhisperNamespace()
        self.nlp = SpacyNamespace()
        self.path = PathfindingNamespace()
        self.subcommand: Optional[str]
        self.base_keys = ['stt', 'nlp', 'path', 'subcommand', 'base_keys']

    def normalize(self):
        # Unflatten nested keys (ie keys like 'stt.subkey' -> self.stt.subkey)
        nested_keys = [key for key in self.__dict__.keys() if '.' in key]

        for key in nested_keys:
            category, subkey = key.split('.', 1)
            setattr(getattr(self, category), subkey, getattr(self, key))
            delattr(self, key)

        if self.subcommand:
            unnested_keys = [key for key in self.__dict__.keys() if '.' not in key and key not in self.base_keys]
            for key in unnested_keys:
                setattr(getattr(self, self.subcommand), key, getattr(self, key))
                delattr(self, key)

class AYA:
    def __init__(self, args: AYAParameters):
        self.args = args
        self._whisper: Whisper = None
        self._spacy: Spacy = None
        self._path: Pathfinder = None
        

    @property
    def whisper(self):
        if not self._whisper:
            if not self.has_stt():
                raise ValueError("Whisper is not available when running subcommand %s" % self.args.subcommand)
            self._whisper = Whisper(self.args.stt)
        return self._whisper
    
    @property
    def spacy(self):
        if not self._spacy:
            if not self.has_nlp():
                raise ValueError("Spacy is not available when running subcommand %s" % self.args.subcommand)
            self._spacy = Spacy(self.args.nlp)
        return self._spacy
    
    @property
    def path(self):
        if not self._path:
            if not self.has_path():
                raise ValueError("Path is not available when running subcommand %s" % self.args.subcommand)
            self._path = Pathfinder(self.args.path)
        return self._path

    def has_stt(self):
        """Return True if the STT feature is required (either as main feature or as subcommand)"""
        return (not self.args.subcommand and self.args.nlp.use_stt) or self.args.subcommand == 'stt'
    
    def has_nlp(self):
        """Return True if the NLP feature is required (either as main feature or as subcommand)"""
        return not self.args.subcommand or self.args.subcommand == 'nlp'
    
    def has_path(self):
        """Return True if the pathfinding feature is required (either as main feature or as subcommand)"""
        return not self.args.subcommand or self.args.subcommand == 'path'

    def validate(self):
        if self.has_stt() and not self.args.stt.validate(self.args.subcommand == 'stt'):
            return False
        if self.has_nlp() and not self.args.nlp.validate(self.args.subcommand == 'nlp'):
            return False
        return True
    
    def on_transcript(self, transcript: str):
        """We received a transcript from the STT engine"""
        logging.debug('STT Transcription: %s', transcript)
        if self.has_nlp():
            response = self.spacy.process_sentence(transcript)
            self.on_trip_response(response)
        return True
    
    def on_trip_response(self, response: 'TripResponse'):
        """We received a trip response from the NLP engine"""
        logging.debug('NLP Transcription: %s' % response.to_csv_line(with_source=True))
        if response.is_trip and self.has_path() and not self.args.nlp.no_pathfinding:
            if not response.trip.departure or not response.trip.arrival:
                return
            self.path.compute_path(response.trip.departure, response.trip.arrival)
        return True
    
    def run_stt(self):
        return self.whisper.run(self.on_transcript)

    def run_nlp(self):
        return self.spacy.run(self.on_trip_response)
    
    def run_path(self):
        return self.path.run()
    
    def run(self):
        if not self.validate():
            return -1
        if self.args.subcommand:
            if self.args.subcommand == 'stt':
                return self.run_stt()
            elif self.args.subcommand == 'nlp':
                return self.run_nlp()
            elif self.args.subcommand == 'path':
                return self.run_path()
        else:
            if self.args.nlp.use_stt:
                self.run_stt()
            else:
                self.run_nlp()
        return 0


def main():
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] - %(name)s: %(message)s")
    parser = get_arg_parser()
    args: AYAParameters = parser.parse_args(namespace=AYAParameters())
    args.normalize()
    aya = AYA(args)
    return aya.run()

if __name__ == "__main__":
    sys.exit(main())

