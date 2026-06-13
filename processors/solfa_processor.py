from music21 import stream

SOLFA_MAP = {
    1: "do",
    2: "re",
    3: "mi",
    4: "fa",
    5: "sol",
    6: "la",
    7: "ti"
}


def convert_to_solfa(notes):

    melody_stream = stream.Stream()

    for n in notes:
        melody_stream.append(n)

    detected_key = melody_stream.analyze("key")

    scale = detected_key.getScale()

    solfa = []

    for n in notes:

        degree = scale.getScaleDegreeFromPitch(
            n.pitch
        )

        if degree in SOLFA_MAP:
            solfa.append(
                SOLFA_MAP[degree]
            )

    cleaned = []

    for s in solfa:

        if not cleaned or cleaned[-1] != s:
            cleaned.append(s)

    return {
        "key": str(detected_key),
        "solfa": cleaned
    }
