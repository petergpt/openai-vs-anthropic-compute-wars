from __future__ import annotations

import shutil

import analyze_allocations
import build_openai_anthropic_evidence_pack
import build_openai_anthropic_monthly_visualization
import build_openai_anthropic_open_data_package
import build_openai_anthropic_publishable_view
import build_openai_anthropic_training_story
import paths


def remove_path(path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def clean_generated_outputs() -> None:
    remove_path(paths.DERIVED_DIR)
    remove_path(paths.PACKAGE_DIR)
    remove_path(paths.DOCS_INDEX_HTML)
    remove_path(paths.MONTHLY_VISUALIZATION_HTML)
    remove_path(paths.TRAINING_STORY_HTML)


def main() -> None:
    clean_generated_outputs()
    paths.ensure_generated_dirs()

    analyze_allocations.main()
    build_openai_anthropic_evidence_pack.main()
    build_openai_anthropic_publishable_view.main()
    build_openai_anthropic_monthly_visualization.main()
    build_openai_anthropic_open_data_package.main()
    build_openai_anthropic_training_story.main()


if __name__ == "__main__":
    main()
