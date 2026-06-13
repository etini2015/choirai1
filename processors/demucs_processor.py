import subprocess
import os


def separate_vocals(audio_path):

    output_dir = "separated"

    subprocess.run(
        [
            "python",
            "-m",
            "demucs",
            "--two-stems=vocals",
            audio_path,
            "-o",
            output_dir
        ],
        check=True
    )

    song_name = os.path.splitext(
        os.path.basename(audio_path)
    )[0]

    vocals_path = os.path.join(
        output_dir,
        "htdemucs",
        song_name,
        "vocals.wav"
    )

    return vocals_path
