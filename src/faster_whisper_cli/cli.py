import argparse

from faster_whisper import WhisperModel


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio with faster-whisper."
    )
    parser.add_argument("audio", help="Path to the audio file")
    parser.add_argument(
        "--model",
        "-m",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        default="small",
        help="Model size (default: small)",
    )
    parser.add_argument(
        "--language",
        "-l",
        default=None,
        help="Language code, such as es or en. Omit for automatic detection.",
    )
    parser.add_argument(
        "--device",
        "-d",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to use (default: cuda)",
    )

    args = parser.parse_args()

    compute_type = "float16" if args.device == "cuda" else "int8"
    model = WhisperModel(args.model, device=args.device, compute_type=compute_type)

    segments, info = model.transcribe(
        args.audio,
        beam_size=5,
        language=args.language,
    )

    print(f"Detected language: {info.language} ({info.language_probability:.2f})")
    for segment in segments:
        print(
            f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}", flush=True
        )


if __name__ == "__main__":
    main()
