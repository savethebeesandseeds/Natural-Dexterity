"""Create a synthetic contract fixture; this is not a Gemini voice demonstration."""
from __future__ import annotations

import argparse
from pathlib import Path
import struct
import tempfile
import wave

from web_radio.contract import create_release, validate_package


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/workspace/.runtime/examples/contract-v1")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        print(validate_package(output).model_dump_json(indent=2))
        return
    with tempfile.TemporaryDirectory(prefix="web-radio-contract-example-") as scratch:
        source = Path(scratch) / "tone.wav"
        with wave.open(str(source), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(24000)
            audio.writeframes(b"".join(
                struct.pack("<h", 1200 if (index // 24) % 2 else -1200)
                for index in range(6000)
            ))
        manifest = create_release(
            source, "Synthetic verification tone. This package contains no generated speech.",
            output, programme_id="contract-fixture", title="Synthetic contract fixture",
            kind="ident", release_id="contract-fixture-v1",
        )
    print(manifest.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
