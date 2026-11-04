#!/usr/bin/env python
"""Run the relevance screening pipeline."""
import argparse
import sys
import logging
from pathlib import Path

# Add src/ to Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vidscreen.config import load_settings
from vidscreen.logging_setup import setup_logging
from vidscreen.screening.run import run


def main():
    parser = argparse.ArgumentParser(description="Run relevance screening pipeline")
    parser.add_argument("--stage", choices=["cv", "all"], default="cv",
                        help="cv = cross-validation + model selection; all = CV + final screening labels")
    parser.add_argument("--config", help="Path to settings.yaml")
    parser.add_argument("--no-embeddings", action="store_true",
                        help="Skip embedding features (use TF-IDF only) - faster")
    parser.add_argument("--labeled-path", help="Path to labeled set CSV")
    args = parser.parse_args()

    settings = load_settings(args.config)
    log_path = setup_logging(settings.log_dir, "INFO")
    log = logging.getLogger("vidscreen")
    log.info("Log file: %s", log_path)
    log.info("Starting screening: stage=%s, embeddings=%s", args.stage, not args.no_embeddings)

    report = run(
        settings,
        stage=args.stage,
        use_embeddings=not args.no_embeddings,
        labeled_path=Path(args.labeled_path) if args.labeled_path else None,
    )

    log.info("Screening complete.")
    log.info("Report: %s", settings.final_dir / "screening" / "screening_report.md")


if __name__ == "__main__":
    main()