import subprocess
import os


def separate_vocals(audio_path):

    output_dir = "./choirai_pipeline_vault/separated"
    os.makedirs(output_dir, exist_ok=True)

    command = [
        "python",
        "-m",
        "demucs",
        "--two-stems=vocals",
        "-n",
        "htdemucs_light",

        # 🔥 IMPORTANT FIX: avoids torchcodec crash
        "--audio-backend=soundfile",

        "-o",
        output_dir,

        audio_path
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    # ---- DEBUG LOGS (VERY IMPORTANT) ----
    print("=== DEMUCS STDOUT ===")
    print(result.stdout)

    print("=== DEMUCS STDERR ===")
    print(result.stderr)

    # ---- FAIL FAST IF ERROR ----
    if result.returncode != 0:
        raise Exception(
            f"Demucs failed:\n{result.stderr}"
        )

    # ---- GET SONG NAME ----
    song_name = os.path.splitext(
        os.path.basename(audio_path)
    )[0]

    # ---- PATH TO VOCALS ----
    vocals_path = os.path.join(
        output_dir,
        "htdemucs_light",
        song_name,
        "vocals.wav"
    )

    if not os.path.exists(vocals_path):
        raise FileNotFoundError(
            f"Vocals file not found at: {vocals_path}"
        )

    return vocals_path
