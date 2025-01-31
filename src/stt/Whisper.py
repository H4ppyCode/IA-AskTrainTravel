from typing import Callable, Optional
import argparse
import glob
import logging
import sys
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from queue import Queue
from time import sleep

import numpy as np
import torch
import whisper
import speech_recognition as sr

class WhisperModel(Enum):
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    TURBO = "turbo"
    
    def __str__(self) -> str:
        return self.value


class WhisperNamespace(argparse.Namespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model: WhisperModel
        self.audio_files: str
        self.sentences_limit: Optional[int]
        self.mic_threshold: int
        self.record_timeout: float
        self.sentence_timeout: float
        self.microphone: str
        self.sample_rate: int

    def validate(self, is_runned_as_subcommand: bool) -> bool:
        if 'linux' in sys.platform:
            if self.microphone.lower() and self.microphone.lower() == 'list':
                print("Available microphone devices are: ")
                for name in sr.Microphone.list_microphone_names():
                    print(f"- \"{name}\"")
                return False
            # We need a valid audio source to record from if we're not using audio files.
            elif not self.audio_files:
                found = False
                for name in sr.Microphone.list_microphone_names():
                    if self.microphone.lower() in name.lower():
                        found = True
                        break
                if not found:
                    logging.error(f"Microphone \"{self.microphone}\" not found. Run with 'list' as microphone name to list available microphones.")
                    return False
        return True



def fill_whisper_parser(parser: argparse.ArgumentParser, args_prefix: str = "") -> None:
    parser.add_argument(f"--{args_prefix}model", default=WhisperModel.SMALL, help="Whisper model to use. Default: small",
                        choices=[WhisperModel.TINY, WhisperModel.BASE, WhisperModel.SMALL, WhisperModel.MEDIUM, WhisperModel.LARGE, WhisperModel.TURBO], type=WhisperModel)
    
    # STT Local audio files parameters
    parser.add_argument(f"--{args_prefix}audio-files", default=None, help="Glob pattern for audio files", type=str)
    # STT Live recording parameters
    parser.add_argument(f"--{args_prefix}sentences-limit", default=None, help="Limit of sentences to record. Only stop with Ctrl+C if not set.", type=int)
    parser.add_argument(f"--{args_prefix}mic-threshold", default=1000,
                        help="Microphone detection threshold to start a sentence.", type=int)
    parser.add_argument(f"--{args_prefix}record-timeout", default=2,
                        help="Max whisper phrase parts length in seconds. See https://github.com/Uberi/speech_recognition/blob/master/reference/library-reference.rst#recognizer_instancelistensource-audiosource-timeout-unionfloat-none--none-phrase_time_limit-unionfloat-none--none-snowboy_configuration-uniontuplestr-iterablestr-none--none---audiodata.", type=float)
    parser.add_argument(f"--{args_prefix}sentence-timeout", default=3,
                        help="After how many seconds should we consider the sentence is over, and start a new line in the transcription.", type=float)
    parser.add_argument(f"--{args_prefix}sample-rate", default=16000, help="Microphone sample rate. Default to 16000Hz.", type=int)
    
    if 'linux' in sys.platform:
        mic_help = "Microphone name to listen to."
        if not args_prefix:
            mic_help += "Run this with 'list' to view available microphones."
        parser.add_argument(f"--{args_prefix}microphone", default='pulse', help=mic_help, type=str)

class Whisper:
    def __init__(self, config: WhisperNamespace):
        self.config = config
        self._model: whisper.Whisper = None
        # Live record properties
        self._audio_recorder: sr.Recognizer = None
        self._audio_source: sr.Microphone = None

    @property
    def model(self):
        if not self._model:
            self._model = whisper.load_model(str(self.config.model))
        return self._model
    
    @property
    def audio_source(self):
        if not self._audio_source:
            if 'linux' in sys.platform:
                mic_name = self.config.microphone
                if not mic_name or mic_name == 'list':
                    print("Available microphone devices are: ")
                    for index, name in enumerate(sr.Microphone.list_microphone_names()):
                        print(f"- \"{name}\"")
                    return
                else:
                    for index, name in enumerate(sr.Microphone.list_microphone_names()):
                        if mic_name.lower() in name.lower():
                            self._audio_source = sr.Microphone(sample_rate=self.config.sample_rate, device_index=index)
                            break
            else:
                self._audio_source = sr.Microphone(sample_rate=self.config.sample_rate)
        return self._audio_source

    
    @property
    def audio_recorder(self):
        if not self._audio_recorder:
            # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
            self._audio_recorder = sr.Recognizer()
            self._audio_recorder.energy_threshold = self.config.mic_threshold
            # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
            self._audio_recorder.dynamic_energy_threshold = False

            with self.audio_source:
                self._audio_recorder.adjust_for_ambient_noise(self.audio_source)
        return self._audio_recorder


    def transcript_audio_file(self, file_name: str):
        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.load_audio(file_name)
        audio = whisper.pad_or_trim(audio)

        # make log-Mel spectrogram and move to the same device as the model
        # only for large model
        # mel = whisper.log_mel_spectrogram(audio=audio, n_mels=128).to(model.device)

        mel = whisper.log_mel_spectrogram(audio=audio).to(self.model.device)

        start = datetime.now()
        # detect the spoken language
        _, probs = self.model.detect_language(mel)

        lang = max(probs, key=probs.get)

        logging.debug(f"Detected language: {lang}")

        if (lang != "fr" ):
            # TODO I think there is specific error to return in the PDF
            logging.warning("Language not supported")
            return None

        # decode the audio
        options = whisper.DecodingOptions()
        result = whisper.decode(self.model, mel, options)

        logging.debug(f"Decoding took: {datetime.now() - start}")
        return result.text

    def live_record(self, on_transcript: Callable[[str], bool] = None):
        # The last time a recording was retrieved from the queue.
        phrase_time = None
        # Thread safe Queue for passing data from the threaded recording callback.
        data_queue = Queue()
        sentences = ['']
        nb_sentences = 0

        def record_callback(_, audio:sr.AudioData) -> None:
            """
            Threaded callback function to receive audio data when recordings finish.
            audio: An AudioData containing the recorded bytes.
            """
            # Grab the raw bytes and push it into the thread safe queue.
            data = audio.get_raw_data()
            data_queue.put(data)

        # Create a background thread that will pass us raw audio bytes.
        # We could do this manually but SpeechRecognizer provides a nice helper.
        self.audio_recorder.listen_in_background(self.audio_source, record_callback, phrase_time_limit=self.config.record_timeout)

        # Cue the user that we're ready to go.
        logging.info("Model loaded. Start live audio recording\n")

        while not self.config.sentences_limit or nb_sentences < self.config.sentences_limit:
            try:
                now = datetime.utcnow()
                # Pull raw recorded audio from the queue.
                if not data_queue.empty():
                    phrase_complete = False
                    # If enough time has passed between recordings, consider the phrase complete.
                    # Clear the current working audio buffer to start over with the new data.
                    if phrase_time and now - phrase_time > timedelta(seconds=self.config.sentence_timeout):
                        phrase_complete = True
                    # This is the last time we received new audio data from the queue.
                    phrase_time = now
                    
                    # Combine audio data from queue
                    audio_data = b''.join(data_queue.queue)
                    data_queue.queue.clear()
                    
                    # Convert in-ram buffer to something the model can use directly without needing a temp file.
                    # Convert data from 16 bit wide integers to floating point with a width of 32 bits.
                    # Clamp the audio stream frequency to a PCM wavelength compatible default of 32768hz max.
                    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0


                    # TODO detect language
                    # mel = whisper.log_mel_spectrogram(audio_np).to(audio_model.device)
                    # _, probs = audio_model.detect_language(mel)
                    # lang = max(probs, key=probs.get)
                    # print(f"Detected language: {lang}")
                    
                    # Read the transcription.
                    result = self.model.transcribe(audio_np, fp16=torch.cuda.is_available())
                    text = result['text'].strip()

                    # If we detected a pause between recordings, add a new item to our sentences.
                    # Otherwise edit the existing one.
                    if phrase_complete:
                        nb_sentences += 1
                        # If we have a callback and it returned True, we consider the sentence as handled and therefore we don't add it to the list of sentences.
                        if not (on_transcript and on_transcript(text)):
                            sentences.append(text)
                        else:
                            continue
                    else:
                        sentences[-1] = text
                else:
                    # Infinite loops are bad for processors, let it rest.
                    sleep(0.25)
            except KeyboardInterrupt:
                break

        return sentences
    
    def run(self, on_transcript: Callable[[str], bool] = None):
        if self.config.audio_files:
            logging.info("Transcripting files matching glob pattern: '%s'" % self.config.audio_files)
            for file in glob.glob(self.config.audio_files):
                logging.info("Transcripting file %s" % file)
                transcription = self.transcript_audio_file(file)
                if on_transcript:
                    on_transcript(transcription)
                else:
                    logging.info("Transcription: %s" % transcription)
            logging.info("No more files to transcript")
            return 0
        else:
            logging.info("Running STT on microphone. Press Ctrl+C to stop.")
            self.live_record(on_transcript=on_transcript)
            logging.info("End of live record")
            return 0


def main():
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] - %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
                    prog='Whisper Speech to Text',
                    description='Convert audio files or live recordings to text')
    fill_whisper_parser(parser)
    args: WhisperNamespace = parser.parse_args()
    wth = Whisper(args)
    wth.run()

if __name__ == '__main__':
    main()
