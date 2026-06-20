# Releasing OLM

This checklist is for maintainers preparing a public GitHub and PyPI release.

## Before Publishing

Confirm the version is aligned:

```bash
python - <<'PY'
import pathlib, re

pyproject = pathlib.Path("pyproject.toml").read_text()
init = pathlib.Path("src/olm/__init__.py").read_text()

project_version = re.search(r'^version = "([^"]+)"', pyproject, re.M).group(1)
package_version = re.search(r'__version__ = "([^"]+)"', init).group(1)

assert project_version == package_version
print(project_version)
PY
```

Run the release checks:

```bash
python -m compileall -q src tests scripts
pytest -q tests

cd website
npm run lint
npm run build
cd ..

rm -rf dist build
python -m build
twine check dist/*
```

## GitHub Release

1. Push the release commit to `main`.
2. Confirm GitHub Pages builds successfully.
3. Create an annotated tag:

```bash
git tag -a v2.2.0 -m "OpenLanguageModel v2.2.0"
git push origin v2.2.0
```

4. Draft a GitHub release for `v2.2.0`.
5. Use [`release-v2.2.0.md`](release-v2.2.0.md) as the release body.
6. Publish the release.

## PyPI

The repository includes a GitHub Actions workflow that publishes to PyPI when a GitHub release is published. Before publishing the GitHub release, configure PyPI Trusted Publishing for:

- PyPI project: `openlanguagemodel`
- Owner/repository: `openlanguagemodel/openlanguagemodel`
- Workflow: `publish.yml`
- Environment: leave unset unless the workflow is later changed to require one

After the release workflow completes, verify:

```bash
python -m pip index versions openlanguagemodel
python -m pip install --upgrade openlanguagemodel==2.2.0
python - <<'PY'
import olm
print(olm.__version__)
PY
```

If trusted publishing is not configured yet, do not fall back to a local upload unless the package owner explicitly approves it.
