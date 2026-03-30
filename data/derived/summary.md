# Primary User Compute Allocation

## Method

- Current snapshot is taken as of 2026-03-27 using the latest timeline row on or before that date.
- Year-end snapshots use the latest timeline row on or before December 31 of each year.
- Capacity is allocated to the listed user when a single user is present.
- When multiple users are listed, capacity is assigned to the first-listed user unless the notes support an exact split.
- Centers with no listed user remain in the `Unresolved` bucket rather than being forced onto an owner.

## Current Snapshot

| Company | H100 eq | Power MW | Centers |
| --- | --- | --- | --- |
| Anthropic | 900262 | 1433.0 | 2 |
| OpenAI | 833945 | 991.0 | 3 |
| Meta | 761066 | 893.0 | 2 |
| xAI | 556634 | 727.0 | 2 |
| Google DeepMind | 382769 | 851.0 | 4 |
| Alibaba | 132895 | 203.0 | 1 |
| Unresolved | 38706 | 44.8 | 1 |
| G42 | 19859 | 19.0 | 1 |

## Year-End H100 Equivalents By Primary User

| Company | 2024 | 2025 | 2026 | 2027 | 2028 | 2029 |
| --- | --- | --- | --- | --- | --- | --- |
| Alibaba | 0 | 66195 | 132895 | 132895 | 132895 | 132895 |
| Anthropic | 0 | 685914 | 1659690 | 2316916 | 2316916 | 2316916 |
| G42 | 0 | 19859 | 71753 | 71753 | 71753 | 71753 |
| Google DeepMind | 174432 | 382769 | 1255595 | 3303113 | 3899373 | 3899373 |
| Meta | 60753 | 325875 | 1379616 | 2860161 | 7013112 | 7013112 |
| Microsoft | 0 | 0 | 159552 | 159552 | 478655 | 638207 |
| OpenAI | 29151 | 833945 | 2778656 | 7719574 | 9488144 | 13343627 |
| Unresolved | 0 | 0 | 1025439 | 3673203 | 3673203 | 3673203 |
| xAI | 100000 | 553714 | 1665387 | 1665387 | 1665387 | 1665387 |

## Key Issues

- High-severity issues: 3
- Medium-severity issues: 12
- Current unresolved capacity: 38706 H100 eq across 1 centers.
- `Fluidstack Lake Mariner` is split exactly by building across G42 and Anthropic because the notes provide a user-by-building mapping.
- `Microsoft Fairwater Wisconsin` is allocated entirely to OpenAI because the dataset lists `OpenAI` first in a multi-user field, but this remains a material ambiguity.
- `xAI Colossus 1` has a non-monotonic timeline, so trend work should use latest snapshots rather than cumulative maxima.

## Output Files

- `data/derived/data_center_primary_user_mapping.csv`
- `data/derived/data_center_allocations_by_snapshot.csv`
- `data/derived/company_capacity_by_snapshot.csv`
- `data/derived/allocation_issues.csv`
