from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH


def transcribe_to_midi(vocal_file):

    model_output, midi_data, note_events = predict(
        audio_path=vocal_file,
        model_or_model_path=ICASSP_2022_MODEL_PATH
    )

    return midi_data
