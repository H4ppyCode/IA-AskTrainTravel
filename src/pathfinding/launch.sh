python3 ../whisper/main.py ../whisper/nantes-paris.mp3 | tail -n -1 | python3 ../spaCy/nlp_processor.py | paste -d ' ' - - | xargs  python3 main.py
