from music21 import converter


def extract_melody(midi_path):

    score = converter.parse(midi_path)

    melody = {}

    for n in score.flatten().notes:

        if not hasattr(n, "pitch"):
            continue

        offset = round(float(n.offset), 2)

        if offset not in melody:
            melody[offset] = n

        elif n.pitch.midi > melody[offset].pitch.midi:
            melody[offset] = n

    return list(melody.values())
