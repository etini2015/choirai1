import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Reduce logging noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # Mute standard TensorFlow CPU warnings

import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from basic_pitch.inference import predict_and_save
from music21 import converter

app = FastAPI(
    title="Sol-fa Transcription API",
    description="Converts raw audio streams directly into Tonic Sol-fa syllables"
)

# Enable CORS so your Lovable.dev web application can securely fetch data
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your Lovable domain URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Standard Sol-fa syllables mapped to numerical scale degrees
SOLFA_MAP = {
    1: "do",
    2: "re",
    3: "mi",
    4: "fa",
    5: "sol",
    6: "la",
    7: "ti"
}

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "Sol-fa Transcriber Engine"}

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    # 1. Validate file extensions
    allowed_extensions = [".wav", ".mp3", ".ogg", ".flac", ".m4a"]
    file_ext = os.path.splitext(file.filename).lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Please upload: {', '.join(allowed_extensions)}"
        )

    # 2. Define temporary operational file paths
    temp_dir = "./temp_processing"
    os.makedirs(temp_dir, exist_ok=True)
    
    input_audio_path = os.path.join(temp_dir, f"upload_{file.filename}")
    output_midi_dir = os.path.join(temp_dir, "midi_out")
    os.makedirs(output_midi_dir, exist_ok=True)
    
    try:
        # 3. Save incoming stream to local storage
        with open(input_audio_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 4. Transcribe audio straight to absolute MIDI using correct function name
        predict_and_save(
            audio_path_list=[input_audio_path],
            output_directory=output_midi_dir,
            save_midi=True,
            sonify_midi=False,
            save_model_outputs=False,
            save_notes=False
        )
        
        # 5. SAFE LOOKUP: Locate the generated MIDI file
        if not os.path.exists(output_midi_dir):
            raise HTTPException(status_code=500, detail="Midi output directory was not created.")
            
        generated_files = os.listdir(output_midi_dir)
        midi_files = [f for f in generated_files if f.endswith('.mid')]
        
        if not midi_files:
            raise HTTPException(
                status_code=500, 
                detail=f"Transcription completed but no MIDI files found. Contents: {generated_files}"
            )
        
        # Safely target the first file string directly out of the list
        midi_path = os.path.join(output_midi_dir, midi_files[0])
        
        # 6. Parse absolute pitches using musicology frameworks
        score = converter.parse(midi_path)
        detected_key = score.analyze('key')
        
        # 7. Flatten note layers and translate to relative scale degrees
        solfa_sequence = []
        for note in score.flat.notes:
            # If the output contains chords, pick the highest root pitch
            pitch_to_analyze = note.pitches[-1] if hasattr(note, 'pitches') else note.pitch
            
            degree = detected_key.getScaleDegreeAndAccidental(pitch_to_analyze)
            
            # Match degree integers or objects to text syllables
            if isinstance(degree, tuple) and len(degree) > 0 and degree[0] in SOLFA_MAP:
                solfa_sequence.append(SOLFA_MAP[degree[0]])
            elif isinstance(degree, int) and degree in SOLFA_MAP:
                solfa_sequence.append(SOLFA_MAP[degree])

        # 8. Compile payload response
        return {
            "success": True,
            "detected_key": f"{detected_key.tonic.name} {detected_key.mode}",
            "solfa": solfa_sequence,
            "note_count": len(solfa_sequence)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Transcription Error: {str(e)}")
        
    finally:
        # 9. Clean up temporary operational file paths from server
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
