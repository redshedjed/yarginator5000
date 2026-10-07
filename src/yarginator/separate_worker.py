"""One separation pass, run by the GPU environment's Python (see separate.py).

Standalone on purpose: that environment has audio-separator + torch-directml but not yarginator
(its numpy 2 would break TensorFlow in the main environment). Arguments come as one JSON string;
the output file names go back on stdout after a marker line. Progress bars go to stderr.
"""
import json
import logging
import sys

MARKER = "@@YARGINATOR-RESULT@@"


def main() -> int:
    a = json.loads(sys.argv[1])
    from audio_separator.separator import Separator
    sep = Separator(log_level=logging.WARNING, model_file_dir=a["model_dir"], output_dir=a["out_dir"],
                    output_format="FLAC", normalization_threshold=1.0, use_directml=a.get("directml", True))
    sep.load_model(model_filename=a["model"])
    files = sep.separate(a["src"])
    print(MARKER + json.dumps([str(f) for f in files]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
