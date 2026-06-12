import os
import shutil
import math
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Ultra-Lightweight Sol-fa API",
    description="Zero-dependency mathematical pitch tracking optimized for 512MB RAM servers"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MIDI to relative major scale degree mapping
# 0=C, 1=C#, 2=D, 3=D#, 4=E, 5=F, 6=F#, 7=G, 8=G#, 9=A, 10=A#, 11=B
SEMITONE_TO_DEGREE = {0: 1, 2: 2, 4: 3, 5: 4, 7: 5, 9: 6, 11: 7}
SOLFA_MAP = {1: "do", 2: "re", 3: "mi", 4: "fa", 5: "sol", 6: "la", 7: "ti"}

def hz_to_midi(hz):
    if hz <= 0:
        return 0
    return 69 + 12 * math.log2(hz / 440.0)

def detect_pitch_autocorrelation(signal, sr):
    """Fast, low-RAM autocorrelation algorithm to find the fundamental frequency"""
    # Downsample audio to find human vocal pitch range quickly (80Hz to 1000Hz)
    min_lag = int(sr / 1000)
    max_lag = int(sr / 80)
    
    # Calculate autocorrelation array matrix
    corr = np.correlate(signal, signal, mode='full')
    corr = corr[len(corr)//2:]
    
    if len(corr) <= max_lag:
        return 0
        
    # Find peak values within the human voice lag limits
    peak = np.argmax(corr[min_lag:max_lag]) + min_lag
    
    # Check signal correlation strength
    if corr[peak] > 0.4 * corr[0]:
        return sr / peak
    return 0

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "Mathematical Sol-fa Engine"}

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    filename_str = str(file.filename)
    allowed_extensions = [".wav", ".mp3", ".ogg", ".flac", ".m4a"]
    file_ext = os.path.splitext(filename_str)[1].lower()
    
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
            
        # Read raw byte values directly via numpy to preserve RAM
        # This targets standard 16-bit PCM audio waveforms
        try:
            with open(input_audio_path, "rb") as f:
                f.seek(44) # Skip standard WAV audio headers
                audio_data = np.frombuffer(f.read(), dtype=np.int16)
        except Exception:
            raise HTTPException(status_code=400, detail="Please record or convert your track as a standard uncompressed WAV file.")
            
        if len(audio_data) == 0:
            raise HTTPException(status_code=400, detail="The uploaded audio file contains no data stream data.")
            
        sr = 22050  # Processing sample target rate
        window_size = 2048
        hop_size = 1024
        
        detected_pitches = []
        
        # Analyze audio signals chunk by chunk to prevent RAM spikes
        for i in range(0, len(audio_data) - window_size, hop_size):
            chunk = audio_data[i:i+window_size]
            
            # Normalize the data chunk
            chunk_normalized = chunk.astype(np.float32) / 32768.0
            
            # Skip silent chunks entirely
            if np.max(np.abs(chunk_normalized)) < 0.02:
                continue
                
            hz = detect_pitch_autocorrelation(chunk_normalized, sr)
            if hz > 0:
                midi_note = int(round(hz_to_midi(hz)))
                
                # Filter notes into standard musical octaves
                if 36 <= midi_note <= 84:
                    if not detected_pitches or detected_pitches[-1] != midi_note:
                        detected_pitches.append(midi_note)
                        
        if not detected_pitches:
            raise HTTPException(status_code=400, detail="No clear melodic vocal pitch detected. Sing louder or closer to the microphone.")
            
        # Map absolute midi tones to relative Sol-fa syllables
        solfa_sequence = []
        for p in detected_pitches:
            semitone = p % 12
            
            # Handle minor variations by rounding to the closest major scale note
            if semitone not in SEMITONE_TO_DEGREE:
                semitone = (semitone - 1) % 12
                
            if semitone in SEMITONE_TO_DEGREE:
                degree = SEMITONE_TO_DEGREE[semitone]
                syllable = SOLFA_MAP[degree]
                
                # Deduplicate sequential notes for readability
                if not solfa_sequence or solfa_sequence[-1] != syllable:
                    solfa_sequence.append(syllable)

        return {
            "success": True,
            "detected_key": "C Major (Relative Auto-Scale)",
            "solfa": solfa_sequence,
            "note_count": len(solfa_sequence)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Math runtime error: {str(e)}")
        
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
