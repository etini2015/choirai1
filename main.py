import os
import shutil
import tempfile
import time

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from processors.demucs_processor import separate_vocals
from processors.basic_pitch_processor import transcribe_to_midi
from processors.melody_extractor import extract_melody
from processors.solfa_processor import convert_to_solfa

app = FastAPI(
    title="ChoirAI Labs API",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "service": "ChoirAI Labs"
    }


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...)
):

    start_time = time.time()

    temp_dir = tempfile.mkdtemp()

    try:

        print("========== STARTING PIPELINE ==========")

        audio_path = os.path.join(
            temp_dir,
            file.filename
        )

        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(
                file.file,
                buffer
            )

        print(f"Audio saved: {audio_path}")

        # -----------------------------------
        # STEP 1: DEMUCS
        # -----------------------------------

        demucs_start = time.time()

        print("STEP 1: Starting Demucs")

        vocals_path = separate_vocals(
            audio_path
        )

        demucs_time = time.time() - demucs_start

        print(
            f"STEP 1 COMPLETE: Demucs finished in "
            f"{demucs_time:.2f} seconds"
        )

        print(f"Vocals path: {vocals_path}")

        # -----------------------------------
        # STEP 2: BASIC PITCH
        # -----------------------------------

        bp_start = time.time()

        print("STEP 2: Starting Basic Pitch")

        midi_data = transcribe_to_midi(
            vocals_path
        )

        bp_time = time.time() - bp_start

        print(
            f"STEP 2 COMPLETE: Basic Pitch finished in "
            f"{bp_time:.2f} seconds"
        )

        # -----------------------------------
        # STEP 3: SAVE MIDI
        # -----------------------------------

        midi_path = os.path.join(
            temp_dir,
            "output.mid"
        )

        with open(midi_path, "wb") as midi_file:
            midi_data.write(midi_file)

        print(f"MIDI saved: {midi_path}")

        # -----------------------------------
        # STEP 4: MELODY EXTRACTION
        # -----------------------------------

        melody_start = time.time()

        print("STEP 3: Extracting melody")

        melody = extract_melody(
            midi_path
        )

        melody_time = time.time() - melody_start

        print(
            f"STEP 3 COMPLETE: Melody extraction finished in "
            f"{melody_time:.2f} seconds"
        )

        # -----------------------------------
        # STEP 5: SOLFA
        # -----------------------------------

        solfa_start = time.time()

        print("STEP 4: Converting to Solfa")

        result = convert_to_solfa(
            melody
        )

        solfa_time = time.time() - solfa_start

        print(
            f"STEP 4 COMPLETE: Solfa conversion finished in "
            f"{solfa_time:.2f} seconds"
        )

        total_time = time.time() - start_time

        print(
            f"PIPELINE COMPLETE IN "
            f"{total_time:.2f} SECONDS"
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
            ],

            "performance": {
                "demucs_seconds": round(
                    demucs_time, 2
                ),
                "basic_pitch_seconds": round(
                    bp_time, 2
                ),
                "melody_seconds": round(
                    melody_time, 2
                ),
                "solfa_seconds": round(
                    solfa_time, 2
                ),
                "total_seconds": round(
                    total_time, 2
                )
            }
        }

    except Exception as e:

        print(f"PIPELINE ERROR: {str(e)}")

        raise HTTPException(
            status_code=500,
            detail=f"Pipeline Error: {str(e)}"
        )

    finally:

        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )
