# SEU DDS dataset provenance

- Source repository: https://github.com/cathysiyu/Mechanical-datasets
- Source paths: gearbox/bearingset, gearbox/gearset
- Local path: data/seu
- Files: 20 CSV + _files.json + data_manifest_sha256.csv
- Total size: about 1.49 GB
- Manifest restoration verified: 2026-10-18

## Verification method

1. The local `_files.json` records the expected Git blob SHA and size for each CSV.
2. The GitHub API tree for `cathysiyu/Mechanical-datasets` was queried online.
3. For every CSV, path, size and Git blob SHA matched.
4. A local SHA256 manifest was generated at `data/seu/data_manifest_sha256.csv`.

The restored manifests cover 20 CSV files from these immutable official Git trees:
`bearingset` = `7e6e731654541566462d4adcedee04aad150f045`,
`gearset` = `ae2c8c1bc493560794b86cd25771ad7cfe80ebe2`.
Local verification copies were converted from CRLF to LF and matched every
official byte count and Git blob SHA1 before their SHA256 values were recorded.
The source files were not modified. These hashes describe the official LF bytes;
keep CSV line endings unchanged when downloading them.

The provenance audit fails if `_files.json` is missing, an expected entry is
absent, or a CSV hash differs. Missing evidence is never reported as a match.

## Rules

- Treat raw data as read-only.
- Do not modify CSV files.
- Do not create split copies inside this directory; splits belong to protocol execution artifacts.
