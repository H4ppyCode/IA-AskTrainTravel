# DO NOT CALL THIS FILE whisper.py 
import whisper
from datetime import datetime
import sys

def process_file(file_name):
    model = whisper.load_model("small")

    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(file_name)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    # only for large model
    # mel = whisper.log_mel_spectrogram(audio=audio, n_mels=128).to(model.device)

    mel = whisper.log_mel_spectrogram(audio=audio).to(model.device)

    now = datetime.now()
    # detect the spoken language
    _, probs = model.detect_language(mel)

    lang = max(probs, key=probs.get)

    print(f"Detected language: {lang}")

    if (lang != "fr" ):
        # TODO I think there is specific error to return in the PDF
        print("Language not supported")
        exit()


    # decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)

    print(f"Decoding took: {datetime.now() - now}")

    # print the recognized text
    print(result.text)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <file_name>")
        sys.exit(1)

    file_name = sys.argv[1]
    process_file(file_name)

