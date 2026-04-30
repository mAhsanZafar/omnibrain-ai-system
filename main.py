#!/usr/bin/env python3
"""
OmniBrain AI System — Unified Entry Point
==========================================
Launches any component of the system through an interactive menu or a
direct ``--module`` flag.

Usage
-----
  python main.py               # interactive menu
  python main.py --module smai # launch SMAI directly
"""

import argparse
import json
import os
import subprocess
import sys

# Ensure imports from this directory work regardless of cwd
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# Banner
# ─────────────────────────────────────────────────────────────────────────────

BANNER = r"""
╔══════════════════════════════════════════════════════════════════════════╗
║                       OmniBrain AI System                                ║
╠═══════════════════════╦══════════════════════════════════════════════════╣
║  USER INTERFACES      ║  KNOWLEDGE INGESTION                             ║
║  • Jarvis  (voice)    ║  • PDFs_trainer   PDF → NLP classifier           ║
║  • CMAI    (code)     ║  • ebook2pdf      multi-modal QA pipeline        ║
║                       ║  • google_trainer web scrape → training CSV      ║
║  AI CORE / MODELS     ║  • labgen_trainer LibGen → book training data    ║
║  • SMAI (evol. ML)    ╠══════════════════════════════════════════════════╣
║    sklearn auto-sel.  ║  STATIC DATA / CONFIG                            ║
║    DEAP optim.        ║  GS.json · YS.md · ruf.text · model.pkl          ║
║    versioned .pkl     ║  requirements.txt · save_directory.zip           ║
╚═══════════════════════╩══════════════════════════════════════════════════╝
"""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run_script(script_name: str) -> None:
    """Run *script_name* as a subprocess using the current Python interpreter."""
    script_path = os.path.join(BASE_DIR, script_name)
    subprocess.run([sys.executable, script_path])


def _prompt(prompt_text: str, default: str) -> str:
    """Prompt the user; return *default* when the user presses Enter."""
    answer = input(f"  {prompt_text} [{default}]: ").strip()
    return answer if answer else default


# ─────────────────────────────────────────────────────────────────────────────
# Component launchers
# ─────────────────────────────────────────────────────────────────────────────

def launch_jarvis() -> None:
    """
    Voice-controlled desktop assistant (Windows only).
    Launched as a subprocess to avoid importing Windows-only packages
    (win32gui, win32con, pygetwindow) on other platforms.
    """
    print("\n  [Jarvis] Starting voice assistant — Windows only.")
    print("  Make sure a microphone is connected and pyttsx3/pywin32 are installed.")
    _run_script("Jarvis.py")


def launch_cmai() -> None:
    """
    AI coding assistant powered by Salesforce CodeT5.
    Launched as a subprocess because the model (~1.5 GB) is downloaded and
    loaded at module scope, which would block the menu.
    """
    print("\n  [CMAI] Starting CodeT5 coding assistant.")
    print("  The model will be downloaded on the first run (~1.5 GB).")
    _run_script("CMAI.py")


def launch_smai() -> None:
    """
    Self-Modifying AI core: auto-selects the best sklearn classifier from
    CSV + PNG-frame data then optimises hyperparameters with DEAP.
    Paths are prompted here so the user is not limited to the hardcoded
    Windows paths in the original script.
    """
    print("\n  [SMAI] Self-Modifying AI Core")
    print("  ─" * 25)

    default_data  = os.path.join(BASE_DIR, "data")
    default_model = BASE_DIR
    default_frame = os.path.join(BASE_DIR, "frames")

    data_dir  = _prompt("Data directory  (CSV files)", default_data)
    model_dir = _prompt("Model directory (pkl output)", default_model)
    frame_dir = _prompt("Frame directory (PNG files)", default_frame)

    versioning  = _prompt("Enable model versioning?   (y/n)", "y").lower() != "n"
    real_time   = _prompt("Enable real-time training? (y/n)", "n").lower() == "y"

    # Create directories that don't yet exist so SMAI doesn't crash on listdir
    for d in (data_dir, model_dir, frame_dir):
        os.makedirs(d, exist_ok=True)

    from SMAI import SelfModifyingAI  # safe: side-effects are inside __main__

    smai = SelfModifyingAI(
        data_dir, model_dir, frame_dir,
        versioning=versioning,
        real_time_training=real_time,
    )
    smai.run()

    # Interactive prediction loop
    print("\n  SMAI training complete. Enter sample inputs for prediction.")
    while True:
        try:
            user_input = input("  How can I help you? (type 'exit' to quit): ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not user_input or user_input.lower() == "exit":
            break
        try:
            answer = smai.ask_question(user_input)
            print(f"  Prediction: {answer}")
        except Exception as exc:
            print(f"  Prediction error: {exc}")


def launch_ebook2pdf() -> None:
    """
    Multi-modal ingestion + QA pipeline.
    Processes PDFs, EPUBs, audio, images, and video; trains a Keras model;
    builds sentence-transformer embeddings; answers questions via HF QA.
    """
    print("\n  [ebook2pdf] Multi-Modal QA Pipeline")
    print("  A Tkinter folder picker will open to select your training data.")
    from ebook2pdf import main  # safe: side-effects are inside __main__
    main()


def launch_pdfs_trainer() -> None:
    """
    PDF NLP classifier trainer.
    Launched as a subprocess because the original script executes training
    at module scope (no __main__ guard), which would run immediately on import.
    """
    print("\n  [PDFs Trainer] PDF corpus → NLP classifier")
    print("  NOTE: The folder path is currently hardcoded to E:\\BAI\\book in")
    print("        PDFs_trainer.py (line 129). Edit that line to change it.")
    _run_script("PDFs_trainer.py")


def launch_google_trainer() -> None:
    """
    Web scraper: collects title/link pairs from ChatGPT web search and saves
    them as training data.  Loads queries from GS.json when available.
    """
    print("\n  [Google Trainer] Web scraper → training data collector")
    print("  ─" * 25)

    from google_trainer import TrainingDataExtractor  # safe: has __main__ guard

    output_file = _prompt("Output file", "training_data.csv")

    # Attempt to load queries from GS.json (format: GS+=+["query1", ...])
    queries = []  # type: list
    gs_json_path = os.path.join(BASE_DIR, "GS.json")
    if os.path.exists(gs_json_path):
        try:
            with open(gs_json_path, encoding="utf-8") as fh:
                content = fh.read().strip()
            # GS.json starts with "GS+=+[" — strip that prefix to get valid JSON
            if content.startswith("GS+=+["):
                content = content[len("GS+=+"):]  # leaves "[...]"
            raw: list = json.loads(content)
            queries = [q.replace("+", " ") for q in raw if isinstance(q, str)]
            print(f"  Loaded {len(queries)} queries from GS.json.")
        except Exception as exc:
            print(f"  Could not parse GS.json ({exc}). Please enter queries manually.")

    if not queries:
        print("  Enter search queries one per line. Leave blank to finish.")
        while True:
            q = input("  Query > ").strip()
            if not q:
                break
            queries.append(q)

    if not queries:
        print("  No queries provided. Exiting Google Trainer.")
        return

    extractor = TrainingDataExtractor()
    for query in queries:
        encoded = query.replace(" ", "+")
        print(f"  Processing: {query}")
        extractor.scrape_chatgpt(encoded)
        extractor.save_training_data(output_file)
    print(f"\n  Done. Results saved to: {output_file}")


def launch_labgen_trainer() -> None:
    """
    Library Genesis scraper: searches libgen.rs and saves book metadata
    (title, author, content snippet) to a local CSV for use as training data.
    """
    print("\n  [LibGen Trainer] Library Genesis → book training data")
    from labgen_training import main  # safe: has __main__ guard
    main()


def launch_encoding_detector() -> None:
    """
    Utility: detects the character encoding of any local file using chardet.
    Equivalent to testany.py but without its hardcoded path.
    """
    print("\n  [Encoding Detector] Detect file encoding")
    file_path = input("  File path: ").strip()
    if not file_path:
        print("  No path provided.")
        return
    if not os.path.exists(file_path):
        print(f"  File not found: {file_path}")
        return
    try:
        import chardet
        with open(file_path, "rb") as fh:
            raw = fh.read()
        result = chardet.detect(raw)
        print(f"  Encoding   : {result.get('encoding', 'unknown')}")
        print(f"  Confidence : {result.get('confidence', 0):.0%}")
    except ImportError:
        print("  chardet is not installed. Run: pip install chardet")
    except Exception as exc:
        print(f"  Error: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Component registry
# ─────────────────────────────────────────────────────────────────────────────
# Each entry: (cli_key, display_name, short_description, launcher_function)

COMPONENTS = [
    (
        "jarvis",
        "Jarvis Voice Assistant",
        "Voice-controlled desktop assistant (Windows only)",
        launch_jarvis,
    ),
    (
        "cmai",
        "CMAI Coding Assistant",
        "AI code generation REPL powered by Salesforce CodeT5",
        launch_cmai,
    ),
    (
        "smai",
        "SMAI Self-Modifying AI Core",
        "Evolutionary sklearn training + DEAP hyperparameter optimisation",
        launch_smai,
    ),
    (
        "ebook2pdf",
        "ebook2pdf Multi-Modal QA",
        "Ingest PDF/EPUB/audio/video, train Keras model, answer questions",
        launch_ebook2pdf,
    ),
    (
        "pdfs",
        "PDFs Trainer",
        "Train an NLP text classifier from a folder of PDFs",
        launch_pdfs_trainer,
    ),
    (
        "google",
        "Google Web Scraper Trainer",
        "Collect training data by scraping web search results",
        launch_google_trainer,
    ),
    (
        "libgen",
        "LibGen Book Trainer",
        "Scrape book metadata from Library Genesis as training data",
        launch_labgen_trainer,
    ),
    (
        "testany",
        "Encoding Detector",
        "Detect the character encoding of any file (testany.py)",
        launch_encoding_detector,
    ),
]

# Map CLI key → COMPONENTS index for --module flag
_MODULE_MAP = {entry[0]: i for i, entry in enumerate(COMPONENTS)}


# ─────────────────────────────────────────────────────────────────────────────
# Menu
# ─────────────────────────────────────────────────────────────────────────────

def _print_menu() -> None:
    print("\n  Available components:\n")
    for i, (key, name, description, _) in enumerate(COMPONENTS, start=1):
        print(f"  [{i}] {name}  (--module {key})")
        print(f"       {description}")
    print("\n  [0] Exit\n")


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="OmniBrain AI System — unified launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Run without arguments for the interactive menu.\n\n"
            "Available --module values:\n"
            + "\n".join(f"  {key:12s} {name}" for key, name, _, __ in COMPONENTS)
        ),
    )
    parser.add_argument(
        "--module", "-m",
        choices=list(_MODULE_MAP.keys()),
        metavar="MODULE",
        help=(
            "Launch a specific component directly without the interactive menu. "
            f"Choices: {', '.join(_MODULE_MAP)}"
        ),
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    print(BANNER)

    # Non-interactive: launch the requested module and exit
    if args.module:
        idx = _MODULE_MAP[args.module]
        _, name, _, fn = COMPONENTS[idx]
        print(f"  Launching: {name}")
        print("  " + "─" * 50)
        fn()
        return

    # Interactive menu loop
    while True:
        _print_menu()
        try:
            raw = input("  Enter choice: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  Goodbye!\n")
            sys.exit(0)

        if raw == "0":
            print("\n  Goodbye!\n")
            break

        try:
            idx = int(raw) - 1
        except ValueError:
            print(f"\n  '{raw}' is not a valid number. Please try again.")
            continue

        if not (0 <= idx < len(COMPONENTS)):
            print(f"\n  Choice must be between 1 and {len(COMPONENTS)}, or 0 to exit.")
            continue

        _, name, _, fn = COMPONENTS[idx]
        print(f"\n  Launching: {name}")
        print("  " + "─" * 50)
        try:
            fn()
        except KeyboardInterrupt:
            print("\n\n  Interrupted. Returning to menu...")
        except Exception as exc:  # noqa: BLE001
            print(f"\n  Error in {name}: {exc}")

        input("\n  Press Enter to return to the menu...")


if __name__ == "__main__":
    main()
