from __future__ import annotations

import csv

import paths

FLOOR_CSV = paths.COMPANY_CAPACITY_BY_SNAPSHOT_CSV
CSV_OUT = paths.OPENAI_ANTHROPIC_PUBLISHABLE_VIEW_CSV


def load_floor() -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    with FLOOR_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            out[(row["company"], row["snapshot_label"])] = float(row["power_mw"]) / 1000.0
    return out


def fmt(value: float | None) -> str:
    if value is None:
        return ""
    if abs(value - round(value, 1)) < 1e-9:
        return f"{value:.1f}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def build_rows(floor: dict[tuple[str, str], float]) -> list[dict[str, str]]:
    return [
        {
            "year": "2023",
            "metric_unit": "GW",
            "openai_point_estimate": "0.2",
            "openai_low": "0.2",
            "openai_base": "0.2",
            "openai_high": "0.2",
            "openai_type": "official_exact_self_reported_available_compute",
            "openai_site_floor": "",
            "anthropic_point_estimate": "0.08",
            "anthropic_low": "0.05",
            "anthropic_base": "0.08",
            "anthropic_high": "0.15",
            "anthropic_type": "triangulated_range",
            "anthropic_site_floor": "",
            "point_estimate_type": "historical_exact_or_internal_center_of_gravity",
            "summary_note": (
                "OpenAI is from the company memo series. Anthropic is a triangulated range from Google continuity, TPU usage, and CMA evidence of Google/Amazon compute sourcing."
            ),
        },
        {
            "year": "2024",
            "metric_unit": "GW",
            "openai_point_estimate": "0.6",
            "openai_low": "0.6",
            "openai_base": "0.6",
            "openai_high": "0.6",
            "openai_type": "official_exact_self_reported_available_compute",
            "openai_site_floor": fmt(floor[("OpenAI", "year_end_2024")]),
            "anthropic_point_estimate": "0.18",
            "anthropic_low": "0.10",
            "anthropic_base": "0.18",
            "anthropic_high": "0.30",
            "anthropic_type": "triangulated_range",
            "anthropic_site_floor": fmt(floor[("Anthropic", "year_end_2024")]),
            "point_estimate_type": "historical_exact_or_internal_center_of_gravity",
            "summary_note": (
                "Anthropic 2024 is explicitly not zero. The public estimate comes from AWS primary-training-partner evidence and the December 2024 Rainier >5x statement, constrained by later Rainier site scale."
            ),
        },
        {
            "year": "2025",
            "metric_unit": "GW",
            "openai_point_estimate": "1.9",
            "openai_low": "1.9",
            "openai_base": "1.9",
            "openai_high": "1.9",
            "openai_type": "official_exact_self_reported_available_compute",
            "openai_site_floor": fmt(floor[("OpenAI", "year_end_2025")]),
            "anthropic_point_estimate": "1.30",
            "anthropic_low": "1.092",
            "anthropic_base": "1.30",
            "anthropic_high": "1.60",
            "anthropic_type": "floor_plus_platform_uplift_range",
            "anthropic_site_floor": fmt(floor[("Anthropic", "year_end_2025")]),
            "point_estimate_type": "historical_exact_or_internal_center_of_gravity",
            "summary_note": (
                "Anthropic year-end 2025 visible floor is 1.092 GW; public AWS and Google platform evidence imply actual compute access likely exceeded that floor."
            ),
        },
        {
            "year": "2026_current",
            "metric_unit": "GW",
            "openai_point_estimate": "2.0",
            "openai_low": "1.9",
            "openai_base": "2.10",
            "openai_high": "2.40",
            "openai_type": "triangulated_range_above_site_floor",
            "openai_site_floor": fmt(floor[("OpenAI", "current")]),
            "anthropic_point_estimate": "1.6",
            "anthropic_low": "1.433",
            "anthropic_base": "1.80",
            "anthropic_high": "2.30",
            "anthropic_type": "triangulated_range_above_site_floor",
            "anthropic_site_floor": fmt(floor[("Anthropic", "current")]),
            "point_estimate_type": "future_single_estimate_from_debated_range",
            "summary_note": (
                "Current 2026 ranges use the floor as a hard lower bound. OpenAI carries forward the official 1.9 GW 2025 company total as the clean minimum best-view anchor; Anthropic allows a partial Google uplift above the AWS-heavy visible floor."
            ),
        },
        {
            "year": "2026_year_end",
            "metric_unit": "GW",
            "openai_point_estimate": "3.6",
            "openai_low": "3.137",
            "openai_base": "3.80",
            "openai_high": "4.40",
            "openai_type": "triangulated_range_with_floor_low",
            "openai_site_floor": fmt(floor[("OpenAI", "year_end_2026")]),
            "anthropic_point_estimate": "3.3",
            "anthropic_low": "3.0",
            "anthropic_base": "3.60",
            "anthropic_high": "4.80",
            "anthropic_type": "triangulated_range_with_floor_low",
            "anthropic_site_floor": fmt(floor[("Anthropic", "year_end_2026")]),
            "point_estimate_type": "future_single_estimate_from_debated_range",
            "summary_note": (
                "Year-end 2026 ranges start from the heuristic site-backed lower bounds. OpenAI adds only selective non-floor site/provider lanes such as Norway and Cerebras; Anthropic adds a mostly separate Google platform on top of the tracked AWS-heavy floor."
            ),
        },
        {
            "year": "2027_year_end",
            "metric_unit": "GW",
            "openai_point_estimate": "6.2",
            "openai_low": "6.106",
            "openai_base": "7.00",
            "openai_high": "8.00",
            "openai_type": "triangulated_range_with_floor_low",
            "openai_site_floor": fmt(floor[("OpenAI", "year_end_2027")]),
            "anthropic_point_estimate": "6.0",
            "anthropic_low": "3.80",
            "anthropic_base": "6.30",
            "anthropic_high": "7.00",
            "anthropic_type": "triangulated_range_with_floor_low",
            "anthropic_site_floor": fmt(floor[("Anthropic", "year_end_2027")]),
            "point_estimate_type": "future_single_estimate_from_debated_range",
            "summary_note": (
                "Year-end 2027 ranges again start from the lower-bound floor. OpenAI remains sensitive to the full Wisconsin attribution; Anthropic now steps materially higher because the April 6 2026 Google/Broadcom announcement points to a much larger Google TPU lane starting in 2027."
            ),
        },
        {
            "year": "2028_year_end_floor",
            "metric_unit": "GW",
            "openai_point_estimate": "7.2",
            "openai_low": "",
            "openai_base": "",
            "openai_high": "",
            "openai_type": "heuristic_site_backed_lower_bound_only",
            "openai_site_floor": fmt(floor[("OpenAI", "year_end_2028")]),
            "anthropic_point_estimate": "7.0",
            "anthropic_low": "",
            "anthropic_base": "",
            "anthropic_high": "",
            "anthropic_type": "heuristic_site_backed_lower_bound_only",
            "anthropic_site_floor": fmt(floor[("Anthropic", "year_end_2028")]),
            "point_estimate_type": "conservative_forward_estimate_beyond_high_confidence_window",
            "summary_note": (
                "Beyond 2027, the public floor stays incomplete. OpenAI remains a conservative rounded floor proxy, while Anthropic now sits well above its own 2028 5 GW training-run anchor because the April 2026 Google/Broadcom update implies a larger continuing Google lane."
            ),
        },
        {
            "year": "2029_year_end_floor",
            "metric_unit": "GW",
            "openai_point_estimate": "9.0",
            "openai_low": "",
            "openai_base": "",
            "openai_high": "",
            "openai_type": "heuristic_site_backed_lower_bound_only",
            "openai_site_floor": fmt(floor[("OpenAI", "year_end_2029")]),
            "anthropic_point_estimate": "7.6",
            "anthropic_low": "",
            "anthropic_base": "",
            "anthropic_high": "",
            "anthropic_type": "heuristic_site_backed_lower_bound_only",
            "anthropic_site_floor": fmt(floor[("Anthropic", "year_end_2029")]),
            "point_estimate_type": "conservative_forward_estimate_beyond_high_confidence_window",
            "summary_note": (
                "2029 remains overlap-heavy, but a decline would be nonsensical given the higher 2028 anchor and continued River Bend, Google, and Microsoft-linked ramps. Anthropic therefore steps up again above the revised 2028 anchor."
            ),
        },
    ]


def write_csv(rows: list[dict[str, str]]) -> None:
    fieldnames = list(rows[0].keys())
    with CSV_OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)


def build_md(rows: list[dict[str, str]]) -> str:
    return f"""# OpenAI / Anthropic Publishable View

## Validation Standard

- `OpenAI 2023-2025` uses **OpenAI-stated available compute**, not physical datacenter power.
- `Anthropic 2023-2024` uses **triangulated ranges**, because there is no clean public company-total disclosure.
- `Anthropic 2025` uses a **visible site-backed floor plus conservative platform uplift range**.
- `2026 current`, `2026 year-end`, and `2027 year-end` use **future ranges built from the site-backed lower bound plus conservative non-overlapping uplift**.
- `2026 current`, `2026 year-end`, and `2027 year-end` also include a **single best estimate** that is not a midpoint. It is a judgment call that leans toward live or near-live capacity and discounts undelivered headline commitments.
- `2028+` ranges in this file remain **heuristic site-backed lower bounds only**, but the single-number table can carry a conservative forward estimate when a stronger public anchor exists.
- Umbrella platform claims are not summed with site or provider subcomponents.

## Publishable Historical View

| Year | OpenAI range view | Anthropic range view | Label |
| --- | --- | --- | --- |
| 2023 | `0.2 GW` | `0.05-0.15 GW` | OpenAI official exact, Anthropic triangulated range |
| 2024 | `0.6 GW` | `0.10-0.30 GW` | OpenAI official exact, Anthropic triangulated range |
| 2025 | `1.9 GW` | `1.092-1.60 GW` | OpenAI official exact, Anthropic floor-plus-uplift range |
| 2026 current | `1.9-2.40 GW` | `1.433-2.30 GW` | Both triangulated range with hard site-floor low |
| 2026 year-end | `3.137-4.40 GW` | `3.0-4.80 GW` | Both future ranges anchored on lower bounds |
| 2027 year-end | `6.106-8.00 GW` | `3.80-5.80 GW` | Both future ranges anchored on lower bounds |

## Publishable Single-Number View For All Years

| Period | OpenAI best single estimate | Anthropic best single estimate | Why this is the best call |
| --- | --- | --- | --- |
| 2023 | `0.2 GW` | `0.08 GW` | OpenAI is the official company total. Anthropic is the center of gravity of the triangulated range, not a disclosed number. |
| 2024 | `0.6 GW` | `0.18 GW` | OpenAI is the official company total. Anthropic stays clearly above zero but below any 2025-scale interpretation of Rainier. |
| 2025 | `1.9 GW` | `1.3 GW` | OpenAI is the official company total. Anthropic is best read as modestly above the `1.092 GW` visible floor, not equal to the full later platform headlines. |
| 2026 current | `2.0 GW` | `1.6 GW` | Keeps OpenAI anchored to the official `~1.9 GW` 2025 company total, while allowing only modest uplift for Anthropic beyond its live AWS-heavy floor. |
| 2026 year-end | `3.6 GW` | `3.3 GW` | Counts some end-2026 ramp from explicit provider commitments, but not the full nominal value of AWS, Google, Nvidia, Azure, or other umbrella announcements. |
| 2027 year-end | `6.2 GW` | `6.0 GW` | Leaves OpenAI close to the floor because Wisconsin attribution is still the main swing factor; lifts Anthropic materially because the April 2026 Google/Broadcom expansion points to a much larger Google TPU lane starting in 2027. |
| 2028 year-end | `7.2 GW` | `7.0 GW` | OpenAI stays a conservative rounded floor proxy. Anthropic now sits well above its own 2028 `5 GW` training-run anchor because the April 2026 Google/Broadcom update implies a larger continuing Google lane. |
| 2029 year-end | `9.0 GW` | `7.6 GW` | OpenAI still rounds the site-backed lower bound. Anthropic carries the higher 2028 anchor forward with another step-up rather than flattening after the new Google/Broadcom signal. |

Preferred Anthropic midpoints for internal use only:
- `2023`: about `0.08 GW`
- `2024`: about `0.18 GW`
- `2025`: about `1.30 GW`
- `2026 current`: about `1.80 GW`
- `2026 year-end`: about `3.60 GW`
- `2027 year-end`: about `6.30 GW`

Preferred OpenAI midpoints for internal use only:
- `2026 current`: about `2.10 GW`
- `2026 year-end`: about `3.80 GW`
- `2027 year-end`: about `7.00 GW`

## Why This Is Different From The Earlier Floor

- The local site tracker only measures visible operational sites.
- That is why Anthropic showed `0 GW` in 2024 in the floor model: there was no Anthropic-linked site with an operational row by year-end 2024.
- Public sourcing still proves Anthropic had nonzero compute access by then through Google and AWS.
- OpenAI also looked understated in the earlier floor because the company later disclosed a cleaner `0.2 / 0.6 / 1.9 GW` historical series.
- For 2026-2027, the floor remains the anchor, but I now add only conservative uplift that can be defended without stacking umbrella claims.
- For 2026-2027 point estimates, I still lean toward live and near-live capacity rather than headline commitments, but the April 2026 Google/Broadcom update pushes Anthropic's later Google lane far above the earlier conservative call.

## Key Source Logic

- OpenAI:
  - [OpenAI US House Select Cmte Update](https://cdn.openai.com/pdf/045aa967-ee96-4a09-94ee-3098ddf6db2c/OpenAI-US-House-Select-Cmte-Update-%5B021226%5D.pdf)
  - [A business that scales with the value of intelligence](https://openai.com/index/a-business-that-scales-with-the-value-of-intelligence/)
  - [AWS and OpenAI announce multi-year strategic partnership](https://openai.com/index/aws-and-openai-partnership/)
  - [OpenAI, Oracle, and SoftBank expand Stargate with five new AI data center sites](https://openai.com/index/five-new-stargate-sites/)
  - [Introducing Stargate Norway](https://openai.com/index/introducing-stargate-norway/)
  - [OpenAI partners with Cerebras](https://openai.com/index/cerebras-partnership/)
  - [OpenAI and SoftBank Group partner with SB Energy](https://openai.com/index/stargate-sb-energy-partnership/)
- Anthropic continuity and estimate anchors:
  - [Anthropic / Google Cloud partnership](https://www.prnewswire.com/news-releases/anthropic-forges-partnership-with-google-cloud-to-help-deliver-reliable-and-responsible-ai-301738512.html)
  - [Cloud TPU v5e is generally available](https://cloud.google.com/blog/products/compute/announcing-cloud-tpu-v5e-in-ga)
  - [CMA full text decision on Google / Anthropic](https://assets.publishing.service.gov.uk/media/676959bae6ff7c8a1fde9d33/Full_text_decision__.pdf)
  - [Powering the next generation of AI development with AWS](https://www.anthropic.com/news/anthropic-amazon-trainium)
  - [Claude 3.5 Haiku on AWS Trainium2 and model distillation](https://www.anthropic.com/news/trainium2-and-distillation)
  - [AWS activates Project Rainier](https://www.aboutamazon.com/news/aws/aws-project-rainier-ai-trainium-chips-compute-cluster)
  - [AWS Trainium Customers](https://aws.amazon.com/ai/machine-learning/trainium/customers/)
  - [Expanding our use of Google Cloud TPUs and Services](https://www.anthropic.com/news/expanding-our-use-of-google-cloud-tpus-and-services)
  - [Hut 8 reports Q4 and full-year 2025 results](https://br.advfn.com/noticias/PRNUS/2026/artigo/97912535)

## Site-Backed Lower Bounds

| Snapshot | OpenAI site-backed lower bound | Anthropic site-backed lower bound |
| --- | ---: | ---: |
| 2024 year-end | `{rows[1]['openai_site_floor']} GW` | `{rows[1]['anthropic_site_floor']} GW` |
| 2025 year-end | `{rows[2]['openai_site_floor']} GW` | `{rows[2]['anthropic_site_floor']} GW` |
| 2026 current | `{rows[3]['openai_site_floor']} GW` | `{rows[3]['anthropic_site_floor']} GW` |
| 2026 year-end floor | `{rows[4]['openai_site_floor']} GW` | `{rows[4]['anthropic_site_floor']} GW` |
| 2027 year-end floor | `{rows[5]['openai_site_floor']} GW` | `{rows[5]['anthropic_site_floor']} GW` |
| 2028 year-end floor | `{rows[6]['openai_site_floor']} GW` | `{rows[6]['anthropic_site_floor']} GW` |
| 2029 year-end floor | `{rows[7]['openai_site_floor']} GW` | `{rows[7]['anthropic_site_floor']} GW` |

## Files

- `data/derived/openai_anthropic_publishable_view.csv`
"""


def main() -> None:
    paths.DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    floor = load_floor()
    rows = build_rows(floor)
    write_csv(rows)
    print(f"Wrote {CSV_OUT}")


if __name__ == "__main__":
    main()
