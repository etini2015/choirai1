import os
import shutil
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import aubio
from music21 import pitch, key, stream, note

app = FastAPI(
    title="Lightweight Sol-fa Transcription API",
    description="TensorFlow-free monophonic audio transcription engine optimized for low-RAM servers"
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
    return {"status": "healthy", "service": "Lightweight Sol-fa Engine"}

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
            
        # 1. Initialize Aubio's lightweight pitch tracker
        samplerate = 44100
        win_s = 4096   # window size
        hop_s = 512    # hop size
        
        src = aubio.source(input_audio_path, samplerate, hop_s)
        samplerate = src.samplerate
        
        # Uses YinFFT algorithm - highly accurate for monophonic singing/vocal tracks
        pitch_o = aubio.pitch("yinfft", win_s, hop_s, samplerate)
        pitch_o.set_unit("midi")
        pitch_o.set_tolerance(0.8)
        
        detected_pitches = []
        
        # 2. Extract MIDI notes frame by frame
        while True:
            samples, read = src()
            pitch_midi = pitch_o(samples)[0]
            confidence = pitch_o.get_confidence()
            
            # Filter out background silence and unconfident voice glitches
            if pitch_midi > 0 and confidence > 0.85:
                rounded_note = int(round(pitch_midi))
                # Simple de-duplication: avoid stacking the exact same frame pitch
                if not detected_pitches or detected_pitches[-1] != rounded_note:
                    detected_pitches.append(rounded_note)
                    
            if read < hop_s:
                break
                
        if not detected_pitches:
            raise HTTPException(status_code=400, detail="Audio track too quiet or no clear melodic pitches detected.")
            
        # 3. Create a lightweight Music21 stream to analyze key center
        music_stream = stream.Stream()
        for p in detected_pitches:
            n = note.Note()
            n.pitch.midi = p
            music_stream.append(n)
            
        detected_key = music_stream.analyze('key')
        
        # 4. Map pitches to relative Sol-fa syllables
        solfa_sequence = []
        for p in detected_pitches:
            pitch_obj = pitch.Pitch()
            pitch_obj.midi = p
            
            degree = detected_key.getScaleDegreeAndAccidental(pitch_obj)
            if isinstance(degree, tuple) and len(degree) > 0:
                degree = degree[0]
                
            if isinstance(degree, int) and degree in SOLFA_MAP:
                # Basic cleanup: remove consecutive identical notes for cleaner display
                if not solfa_sequence or solfa_sequence[-1] != SOLFA_MAP[degree]:
                    solfa_sequence.append(SOLFA_MAP[degree])

        return {
            "success": True,
            "detected_key": f"{detected_key.tonic.name} {detected_key.mode}",
            "solfa": solfa_sequence,
            "note_count": len(solfa_sequence)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")
        
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
