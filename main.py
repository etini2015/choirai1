import os
import io
import shutil
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
from music21 import converter, key

app = FastAPI(title="Optimized Sol-fa Transcription Engine with Key Override")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SOLFA_MAP = {1: "do", 2: "re", 3: "mi", 4: "fa", 5: "sol", 6: "la", 7: "ti"}

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "Fixed Sol-fa Core"}

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    manual_key: Optional[str] = Query(None, description="Optional override key like 'C', 'G', 'F#m', 'Am'")
):
    filename = str(file.filename)
    
    if not any(filename.lower().endswith(ext) for ext in [".wav", ".mp3", ".m4a", ".ogg"]):
        raise HTTPException(status_code=400, detail="Unsupported audio file format.")

    temp_dir = "./processing_vault"
    os.makedirs(temp_dir, exist_ok=True)
    input_audio_path = os.path.join(temp_dir, f"input_{filename.replace(' ', '_')}")
    
    try:
        with open(input_audio_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Run safely in memory without hitting disk file system blocks
        model_output, midi_data, note_events = predict(
            audio_path=input_audio_path,
            model_or_model_path=ICASSP_2022_MODEL_PATH
        )
        
        midi_stream = io.BytesIO()
        midi_data.write(midi_stream)
        midi_stream.seek(0)
        
        score = converter.parse(midi_stream.read())
        
        # Determine the musical key (use override if provided, otherwise auto-detect)
        if manual_key:
            try:
                detected_key = key.Key(manual_key)
            except Exception:
                raise HTTPException(status_code=400, detail=f"Invalid manual key format: {manual_key}")
        else:
            detected_key = score.analyze('key')
        
        # Get the scale representation safely
        target_scale = detected_key.getScale()
        
        raw_solfa_sequence = []
        
        # --- NEW OPTIMIZATION FILTERING ---
        # Iterate through notes and filter out ultra-short ghost notes or low velocity hums
        for note_obj in score.flat.notes:
            # Skip noise artifacts (notes shorter than 0.15 seconds)
            if note_obj.duration.quarterLength < 0.15:
                continue
                
            if hasattr(note_obj, 'pitch'):
                degree = target_scale.getScaleDegreeFromPitch(note_obj.pitch)
                if degree in SOLFA_MAP:
                    raw_solfa_sequence.append(SOLFA_MAP[degree])
            elif hasattr(note_obj, 'pitches'):
                for p in note_obj.pitches:
                    degree = target_scale.getScaleDegreeFromPitch(p)
                    if degree in SOLFA_MAP:
                        raw_solfa_sequence.append(SOLFA_MAP[degree])
                        break
                        
        # COMPRESSION FILTER: Remove repeating adjacent notes so it looks like readable sheet music
        clean_solfa = []
        for syllable in raw_solfa_sequence:
            if not clean_solfa or clean_solfa[-1] != syllable:
                clean_solfa.append(syllable)

        return {
            "success": True,
            "detected_key": f"{detected_key.tonic.name} {detected_key.mode.capitalize()}",
            "solfa": clean_solfa,
            "note_count": len(clean_solfa)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failure: {str(e)}")
        
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
