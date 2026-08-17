# Contributing to OpenLanguageModel

Thanks for helping make OpenLanguageModel better. OLM is built for people who
want to learn, inspect, and modify language models without losing the ordinary
PyTorch underneath, so contributions that make the code clearer, safer, better
tested, or easier to teach from are especially welcome.

## Ways To Contribute

- Fix docs, examples, typos, broken links, or confusing explanations.
- Add focused tests for model families, training behavior, datasets, or public
  APIs.
- Improve generated API docstrings, shapes, return types, and examples.
- Report bugs with a small reproduction.
- Propose model, training, or documentation improvements through an issue before
  opening a large implementation PR.

## Development Setup

Use Python 3.10, 3.11, or 3.12.

```bash
git clone https://github.com/openlanguagemodel/openlanguagemodel.git
cd openlanguagemodel
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

On Windows, activate the environment with:

```bash
.venv\Scripts\activate
```

Run the Python checks:

```bash
python -m compileall -q src tests scripts
pytest -q tests
```

For website work:

```bash
cd website
npm ci
npm run lint
npm run build
```

## Pull Request Flow

1. Create a branch from `dev` for normal development work.
2. Keep the change focused. Small PRs are easier to review and merge.
3. Add or update tests when behavior changes.
4. Update docs when public APIs, examples, installation, or training behavior
   changes.
5. Open a pull request into `dev`.
6. Fill out the pull request template and link related issues.

`main` is reserved for stable release rollouts. Maintainers merge `dev` into
`main` when preparing a versioned release.

For branch names, use a readable prefix such as:

```bash
git checkout -b tavish/fix-tokenizer-streaming
git checkout -b username/docs-first-model
```

## Code Style

- Follow the style already present in the surrounding file.
- Prefer explicit, readable PyTorch over hidden framework behavior.
- Keep abstractions small and only add them when they remove real repetition or
  clarify a public path.
- Public classes and methods should have useful docstrings, including expected
  tensor shapes and return values where relevant.
- Avoid unrelated refactors inside bug-fix PRs.

## Tests

Please run the relevant subset locally before opening a PR. For broad or public
API changes, run the full suite:

```bash
pytest -q tests
```

For model-family changes, include at least a constructor/config check and a tiny
forward/backward or one-batch training smoke test when practical.

## Documentation

Docs live in `docs/` and are rendered into the website. If you add or change a
public component, update the relevant guide or API docstring. If you add a
notebook, also update `docs/colab-notebooks.md`.

## Adding A Model Family

Before adding a new model family, open an issue describing:

- the model family and reference source
- the architecture pieces needed
- which parts are exact, approximate, or intentionally omitted
- the tests you plan to add

Model implementations should be readable worked examples assembled from OLM's
public components, not hidden configuration blobs.

## Community Expectations

All contributors are expected to follow the
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md). Please report security issues using
[`SECURITY.md`](SECURITY.md), not public issues.
