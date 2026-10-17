# SEU DDS dataset provenance

- Source repository: https://github.com/cathysiyu/Mechanical-datasets
- Source paths: gearbox/bearingset, gearbox/gearset
- Local path: data/seu
- Files: 20 CSV + _files.json + data_manifest_sha256.csv
- Total size: about 1.49 GB
- Verification date: 2026-10-13

## Verification method

1. The local `_files.json` records the expected Git blob SHA and size for each CSV.
2. The GitHub API tree for `cathysiyu/Mechanical-datasets` was queried online.
3. For every CSV, path, size and Git blob SHA matched.
4. A local SHA256 manifest was generated at `data/seu/data_manifest_sha256.csv`.

## Rules

- Treat raw data as read-only.
- Do not modify CSV files.
- Do not create split copies inside this directory; splits belong to protocol execution artifacts.
