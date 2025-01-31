# T-AIA (AYAAAAAAAAAAAAAA)

## For documentation & Work Progress

[![Notion](https://img.shields.io/badge/Notion-000000?style=for-the-badge&logo=notion&logoColor=white)
](https://alpine-cuckoo-e2f.notion.site/T-AIA-901-c6ed78595e584f15847a8320bc7a1f8c)

For the final report, please refer to the [Report](doc/ReportAya.pdf)



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
### OS environnment
You need to install some instance before running 

On Mac OS
```sh
brew install portaudio ffmpeg
```

Other OS, replace 'brew' by your Package Manager

-----------------

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
The expected directory is data/pathfinding.

3. Use our dataset

You can unzip our dataset (a merge from ter and tgv datasets) in data/pathfinding/data.zip.

### Generate dataset for nlp
For nlp_dataset_generator.py, you also need SNCF Datasets to generate the dataset for the NLP model.

`python nlp_dataset_generator.py`

`python nlp_dataset_for_validation.py`

### Test and fine-tune the NLP model
`pip install spacy-lookups-data`

## Usage

The main CLI program can be runned like this:

```sh
python src/AYA.py -h
```

From this you can run each module separately or the whole flow (speech to text -> nlp -> pathfinding).
Some options are available only when the module is runned alone or with the others.
For example, you can't specify an input/output city to the pathfinding module if it takes it inputs from the nlp module.


With the following parameters, you can enter text sentences to test the program. You may have to changed the path to the models.
```sh
python src/AYA.py --nlp.use-stdin --nlp.is-trip-model src/nlp/model_is_trip_trained --nlp.model src/nlp/model_ner_trained --path.output-html data/pathfinding/path.html --path.follow-railways
```

### Speech To Text

To use the speech to text alone you can use `AYA.py` or the underlying stt entrypoint:

```sh
python src/AYA.py stt -h
# Or
python src/stt/Whisper.py -h
```

The speech to text module can either take audio files or do live record:

```sh
# No audio files supplied, try to fallback on microphone (may fail if no default found)
python src/AYA.py stt
# List the available microphones (run the same command then with your mic name)
python src/AYA.py stt --microphone list
# Run STT with audio files
python src/AYA.py stt --audio-files "./audio_files/*.mp3"
```

### Natural Language Processing

As the STT module, you can use the NLP module with `AYA.py` or the underlying nlp entrypoint:

```sh
python src/AYA.py nlp -h
# Or
python src/nlp/Spacy.py -h
```

The NLP module on itself can either read from stdin or text files:

```sh
# No input supplied, will read from stdin
python src/AYA.py nlp
# Explicitly use stdin
python src/AYA.py nlp --use-stdin
# Use input files (in the same format as examples/sample_nlp_input.txt)
python src/AYA.py nlp --input-files "examples/nlp_input*.txt"
```

### Pathfinding

Once again, same usage but with `path`.

```sh
python src/AYA.py path -h
# Or
python src/pathfinding/main.py -h
```

The pathfinding module on itself expects two positional arguments being the departure and destination city.
You should supply --gtfs-path as the folder containing the required GTFS files. (default to 'gtfs')

```sh
# Default configuration
python src/AYA.py path Nantes Paris
```

### All together

When using multiple modules, the module parameters are prefixed by the module name:

- `stt --audio-files myfile` -> `--stt.audio-files myfile`

```sh
# STT->NLP: Run nlp from audio files transcriptions
python src/AYA.py --stt.audio-files "./assets/nantes-paris.mp3" --nlp.no-pathfinding
# STT->NLP: Or from live record (you may have to specify --stt.microphone or tune other parameters)
python src/AYA.py --nlp.no-pathfinding
# NLP->PATH: Takes stdin sentences and forward them to the pathfinder
python src/AYA.py --nlp.use-stdin
# NLP->PATH: Sentences file to pathfinder
python src/AYA.py --nlp.input-files "examples/nlp_input*.txt"

# STT->NLP->PATH: Same as the first two examples but without no-pathfinding (will use live record)
python src/AYA.py
```
