# Contributing

## Development workflow

1. Create a short-lived branch from `main` using `feature/<name>` or `fix/<name>`.
2. Keep commits small and use descriptive messages such as `feat: add temporal split validation`.
3. Run the local quality gate before opening a pull request:

```bash
python -m compileall -q app scripts tests
python -m ruff check app scripts tests --select E9,F63,F7,F82
python -m pytest -q
```

4. Describe the motivation, implementation, tests, and research impact in the pull request.
5. Merge only after the CI workflow passes.

## Scientific software rules

- Do not commit credentials, access tokens, personal data, model caches, or generated databases.
- Record dataset origin, collection parameters, time range, and filters.
- Fix random seeds and report dependency versions for evaluated experiments.
- Use temporal splits when later publications or graph edges could leak into training data.
- Report fallback behavior when an optional model artifact is unavailable.

