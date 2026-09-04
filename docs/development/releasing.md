# Releasing

Releases are automated. Nobody edits a version number by hand.

## The flow

1. PRs land on `main` with [Conventional Commits] prefixes.
2. **release-please** maintains a release PR that accumulates every releasable
   commit since the last tag, with the computed version and changelog.
3. Merging that PR tags `vX.Y.Z` and creates the GitHub release.
4. The tag triggers `publish-wheel` (attaches the wheel to the release) and
   `publish-pypi` (uploads to PyPI).
5. Zenodo's webhook archives the GitHub release and mints a version DOI.

[Conventional Commits]: https://www.conventionalcommits.org/

## What release-please updates

Version bumps propagate from `pyproject.toml` to the archival metadata via
`extra-files` in `release-please-config.json`:

- `.zenodo.json` → `$.version`
- `CITATION.cff` → `$.version`

`tests/test_citation_metadata.py` asserts the three stay in sync, so a bump that
misses one fails CI rather than shipping inconsistent citation metadata.

## Versioning

Pre-1.0, with `bump-minor-pre-major` — `feat:` produces a minor bump, `fix:` and
`perf:` a patch. Housekeeping prefixes are `hidden` and non-releasing.

!!! warning "Only user-visible changes should cut a release"
    Configuring `chore`/`ci`/`test` as releasable produced 0.5.1 and 0.6.1 from
    commits that changed nothing observable. Each such version is a permanent
    artifact — a PyPI release, a git tag, a Zenodo DOI — for a no-op.

## Publishing to PyPI

Uploads use [Trusted Publishing]: PyPI is configured to trust this repository's
`release-please.yml` workflow running in the `pypi` environment, and the job
exchanges a short-lived OIDC token for upload rights. There is no API token
stored anywhere.

[Trusted Publishing]: https://docs.pypi.org/trusted-publishers/

The publish jobs check out `refs/tags/vX.Y.Z` rather than the branch head, so
what is built is exactly what was tagged, and the tag is validated against
`^v[0-9]+\.[0-9]+\.[0-9]+([.-][0-9A-Za-z.-]+)?$` before anything is built.

## Guards

- **Concurrency** is grouped per workflow and ref with `cancel-in-progress:
  false`, so a second push cannot cancel an in-flight publish partway through.
- **The changelog guard** rejects a release whose changelog section is undated
  or still titled "Unreleased" — v0.4.0 shipped that way once.
- **Job-level `permissions` replaces the workflow-level block** rather than
  merging with it. `publish-pypi` needs both `contents: read` (for checkout) and
  `id-token: write` (for OIDC); omitting the first breaks checkout.

## Citation

Cite the **concept DOI** for the software in general, or a **version DOI** for
the specific release you ran. A published result should cite the version DOI and
include the run manifest — one pins the code, the other pins the configuration.
