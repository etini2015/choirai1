import os
import io
import shutil
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Import ONLY the pure in-memory prediction tools and MIDI generators
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
from music21 import converter

app = FastAPI(title="Flag-Free AI Sol-fa Transcription Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SOLFA_MAP = {1: "do", 2: "re", 3: "mi", 4: "fa", 5: "sol", 6: "la", 7: "ti"}

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    filename = str(file.filename)
    
    # Simple validation for audio extensions
    if not any(filename.lower().endswith(ext) for ext in [".wav", ".mp3", ".m4a", ".ogg"]):
        raise HTTPException(status_code=400, detail="Unsupported audio format file.")

    temp_dir = "./processing_vault"
    os.makedirs(temp_dir, exist_ok=True)
    input_audio_path = os.path.join(temp_dir, f"input_{filename.replace(' ', '_')}")
    
    try:
        # 1. Temporarily streaming the incoming audio file safely onto the container
        with open(input_audio_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 2. RUN IN-MEMORY PREDICTION ONLY (No files are saved to disk here)
        # This returns a pure Python object directly in the system RAM
        model_output, midi_data, note_events = predict(
            audio_path=input_audio_path,
            model_or_model_path=ICASSP_2022_MODEL_PATH
        )
        
        # 3. Write MIDI bytes to a virtual in-memory stream buffer instead of a disk file
        midi_stream = io.BytesIO()
        midi_data.write(midi_stream)
        midi_stream.seek(0) # Reset stream pointer to the beginning
        
        # 4. Pass the virtual memory bytes directly into music21
        score = converter.parse(midi_stream.read())
        detected_key = score.analyze('key') # Extract musical key center automatically
        
        solfa_sequence = []
        for note in score.flat.notes:
            if hasattr(note, 'pitch'):
                degree = detected_key.getScaleDegreeAndAccidental(note.pitch)
                if isinstance(degree, tuple):
                    degree = degree[0]
                if degree in SOLFA_MAP:
                    solfa_sequence.append(SOLFA_MAP[degree])
            
            elif hasattr(note, 'pitches'):
                for pitch in note.pitches:
                    degree = detected_key.getScaleDegreeAndAccidental(pitch)
                    if isinstance(degree, tuple):
                        degree = degree[0]
                    if degree in SOLFA_MAP:
                        solfa_sequence.append(SOLFA_MAP[degree])
                        break
                        
        # Deduplicate repeating text blocks for beautiful solfa notation output sheets
        clean_solfa = []
        for syllable in solfa_sequence:
            if not clean_solfa or clean_solfa[-1] != syllable:
                clean_solfa.append(syllable)

        return {
            "success": True,
            "detected_key": f"{detected_key.tonic.name} {detected_key.mode.capitalize()}",
            "solfa": clean_solfa
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Engine Safe Mode Error: {str(e)}")
        
    finally:
        # Safely sweep and clear any leftover tracking paths from the container storage
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
