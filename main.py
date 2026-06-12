import os
import shutil
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import librosa
import scipy
from music21 import pitch, key, stream, note

app = FastAPI(
    title="Data-Science Sol-fa Transcription API",
    description="Lightweight, pre-compiled music analysis pipeline"
)

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
    return {"status": "healthy", "service": "Librosa Sol-fa Engine"}

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    filename_str = str(file.filename)
    allowed_extensions = [".wav", ".mp3", ".ogg", ".flac", ".m4a"]
    file_ext = os.path.splitext(filename_str).lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported format. Please upload: {', '.join(allowed_extensions)}"
        )

    temp_dir = "./temp_processing"
    os.makedirs(temp_dir, exist_ok=True)
    input_audio_path = os.path.join(temp_dir, f"upload_{filename_str.replace(' ', '_')}")
    
    try:
        with open(input_audio_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 1. Load audio matrix into memory safely (downsampled to 22050Hz for speed)
        y, sr = librosa.load(input_audio_path, sr=22050, mono=True)
        
        # 2. Extract fundamental frequency (f0) frames using the YIN algorithm
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        
        detected_pitches = []
        
        # Filter out NaN frames (silence/unvoiced data)
        for freq in f0:
            if not np.isnan(freq) and freq > 0:
                # Convert Hertz frequencies straight to absolute MIDI note integers
                midi_note = int(round(librosa.hz_to_midi(freq)))
                
                # Deduplicate consecutive identical frames for clean arrays
                if not detected_pitches or detected_pitches[-1] != midi_note:
                    detected_pitches.append(midi_note)
                    
        if not detected_pitches:
            raise HTTPException(status_code=400, detail="No clear melodic singing or pitches detected.")
            
        # 3. Stream notes into Music21 to automatically detect the key
        music_stream = stream.Stream()
        for p in detected_pitches:
            n = note.Note()
            n.pitch.midi = p
            music_stream.append(n)
            
        detected_key = music_stream.analyze('key')
        
        # 4. Map absolute MIDI notes to relative Sol-fa syllables
        solfa_sequence = []
        for p in detected_pitches:
            pitch_obj = pitch.Pitch()
            pitch_obj.midi = p
            
            degree = detected_key.getScaleDegreeAndAccidental(pitch_obj)
            if isinstance(degree, tuple) and len(degree) > 0:
                degree = degree
                
            if isinstance(degree, int) and degree in SOLFA_MAP:
                if not solfa_sequence or solfa_sequence[-1] != SOLFA_MAP[degree]:
                    solfa_sequence.append(SOLFA_MAP[degree])

        return {
            "success": True,
            "detected_key": f"{detected_key.tonic.name} {detected_key.mode}",
            "solfa": solfa_sequence,
            "note_count": len(solfa_sequence)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription pipeline error: {str(e)}")
        
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
