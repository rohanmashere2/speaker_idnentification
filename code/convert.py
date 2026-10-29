import subprocess
from pathlib import Path


def convert_to_wav(input_file, sample_rate=16000):
    """
    Convert MP4, M4A, OGG, etc. to 16 kHz mono WAV using FFmpeg.
    """

    input_file = Path(input_file)

    # Output WAV path
    output_file = input_file.with_suffix(".wav")

    # Skip if WAV already exists
    if output_file.exists():
        print(f"Already exists: {output_file}")
        return True

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_file),

        # Disable video
        "-vn",

        # PCM 16-bit WAV
        "-acodec",
        "pcm_s16le",

        # 16 kHz
        "-ar",
        str(sample_rate),

        # Mono
        "-ac",
        "1",

        str(output_file)
    ]

    print(
        f"Converting: {input_file} -> {output_file}"
    )

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode == 0:
            print(
                f"Successfully converted: {output_file}"
            )
            return True

        print(
            f"FFmpeg error for {input_file}:"
        )
        print(result.stderr)

        return False

    except FileNotFoundError:
        print(
            "ERROR: FFmpeg is not installed "
            "or not available in PATH."
        )
        return False

    except Exception as e:
        print(
            f"Conversion error: {e}"
        )
        return False


def scan_and_convert(folder):
    """
    Recursively find MP4, M4A and OGG files
    and convert them to WAV.
    """

    folder = Path(folder)

    if not folder.exists():
        print(
            f"Folder does not exist: {folder}"
        )
        return

    # Supported input formats
    extensions = [
        "*.mp4",
        "*.m4a",
        "*.ogg"
    ]

    files = []

    for extension in extensions:
        files.extend(
            folder.rglob(extension)
        )

    if not files:
        print(
            f"No MP4, M4A or OGG files found "
            f"in {folder}"
        )
        return

    print(
        f"\nFound {len(files)} file(s) "
        f"in {folder}\n"
    )

    successful = 0
    failed = 0

    for input_file in files:

        if convert_to_wav(input_file):
            successful += 1
        else:
            failed += 1

    print(
        "\n========================================"
    )
    print("CONVERSION SUMMARY")
    print("========================================")
    print(f"Total files : {len(files)}")
    print(f"Successful  : {successful}")
    print(f"Failed      : {failed}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("Scanning enrollment folder...")
    scan_and_convert("enroll")

    print("\nScanning test folder...")
    scan_and_convert("test")

    print("\nConversion completed.")