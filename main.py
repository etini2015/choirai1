import os
import shutil
import tempfile

from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import File
from fastapi import HTTPException

from fastapi.middleware.cors import CORSMiddleware

from processors.demucs_processor import (
    separate_vocals
)

from processors.basic_pitch_processor import (
    transcribe_to_midi
)

from processors.melody_extractor import (
    extract_melody
)

from processors.solfa_processor import (
    convert_to_solfa
)

app = FastAPI(
    title="ChoirAI Labs API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health():

    return {
        "status": "healthy",
        "service": "ChoirAI Labs"
    }


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...)
):

    temp_dir = tempfile.mkdtemp()

    try:

        audio_path = os.path.join(
            temp_dir,
            file.filename
        )

        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(
                file.file,
                buffer
            )

        vocals_path = separate_vocals(
            audio_path
        )

        midi_data = transcribe_to_midi(
            vocals_path
        )

        midi_path = os.path.join(
            temp_dir,
            "output.mid"
        )

        with open(midi_path, "wb") as f:
            midi_data.write(f)

        melody = extract_melody(
            midi_path
        )

        result = convert_to_solfa(
            melody
        )

        return {
            "success": True,
            "detected_key": result["key"],
            "solfa": result["solfa"],
            "note_count": len(
                result["solfa"]
            ),
            "debug_notes": [
                n.pitch.nameWithOctave
                for n in melody[:50]
            ]
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )
