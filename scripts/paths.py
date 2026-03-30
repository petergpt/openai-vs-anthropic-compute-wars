from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
REFERENCE_DIR = DATA_DIR / "reference"
DERIVED_DIR = DATA_DIR / "derived"
PACKAGE_DIR = DATA_DIR / "package"
DOCS_DIR = PROJECT_ROOT / "docs"
DOCS_IMAGES_DIR = DOCS_DIR / "images"

DATA_CENTERS_CSV = RAW_DIR / "data_centers.csv"
DATA_CENTER_TIMELINES_CSV = RAW_DIR / "data_center_timelines.csv"
OPENAI_ANTHROPIC_MAJOR_MODEL_RELEASES_SOURCE_CSV = RAW_DIR / "openai_anthropic_major_model_releases.csv"
DATA_CENTER_CHILLERS_CSV = REFERENCE_DIR / "data_center_chillers.csv"
DATA_CENTER_COOLING_TOWERS_CSV = REFERENCE_DIR / "data_center_cooling_towers.csv"

PRIMARY_USER_MAPPING_CSV = DERIVED_DIR / "data_center_primary_user_mapping.csv"
ALLOCATIONS_BY_SNAPSHOT_CSV = DERIVED_DIR / "data_center_allocations_by_snapshot.csv"
COMPANY_CAPACITY_BY_SNAPSHOT_CSV = DERIVED_DIR / "company_capacity_by_snapshot.csv"
ALLOCATION_ISSUES_CSV = DERIVED_DIR / "allocation_issues.csv"
ALLOCATION_SUMMARY_MD = DERIVED_DIR / "summary.md"
OPENAI_ANTHROPIC_EVIDENCE_PACK_CSV = DERIVED_DIR / "openai_anthropic_evidence_pack.csv"
OPENAI_ANTHROPIC_PUBLISHABLE_VIEW_CSV = DERIVED_DIR / "openai_anthropic_publishable_view.csv"
OPENAI_ANTHROPIC_MONTHLY_SERIES_CSV = DERIVED_DIR / "openai_anthropic_monthly_series.csv"
OPENAI_ANTHROPIC_MODEL_OVERLAY_EVENTS_CSV = DERIVED_DIR / "openai_anthropic_model_overlay_events.csv"

DOCS_INDEX_HTML = DOCS_DIR / "index.html"
MONTHLY_VISUALIZATION_HTML = DOCS_DIR / "openai-anthropic-monthly-visualization.html"
TRAINING_STORY_HTML = DOCS_DIR / "openai-anthropic-training-story.html"


def ensure_generated_dirs() -> None:
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
