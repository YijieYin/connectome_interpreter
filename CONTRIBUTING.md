# Contributing

Thanks for considering contributing to `connectome_interpreter`!

## Getting started

```
git clone https://github.com/YijieYin/connectome_interpreter.git
cd connectome_interpreter
pip install -e .
pip install -r requirements.txt -r requirements-types.txt
pip install flake8 black pytest pytest-cov mypy
```

## Making changes

- For small fixes (typos, bugs), feel free to open a PR directly.
- For anything larger — new features, API changes, or behaviour changes — you are encouraged to open an issue first to flag the idea and possibly discuss on direction before you invest time in code.
- Create a branch off `main` for your change.
- Keep pull requests focused — one feature or fix per PR is easier to review.
- Add or update tests in [tests/](tests/) for any behaviour change.
- Add or update docstrings, and update the docs under [docs/](docs/) if you're changing public API or adding a tutorial.

## Before opening a pull request

CI runs the following on every PR (see [.github/workflows/python-app.yml](.github/workflows/python-app.yml)), so it helps to run them locally first:

```
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
black --check -l 88 <changed files>.py   # or just: black -l 88 <changed files>.py
mypy --non-interactive
pytest -v --cov=connectome_interpreter --cov-report=term
```

Notes:
- `black` is only enforced on files changed in the PR (line length 88).
- `mypy` is run but not currently blocking in CI — please still fix new errors where reasonable.

## Reporting bugs / requesting features

Open a [GitHub issue](https://github.com/YijieYin/connectome_interpreter/issues). For dataset requests, feature requests, or general feedback, you can also email `yy432` at `cam.ac.uk`.

## Cell type function list

The [community-curated table of known cell type functions](https://tinyurl.com/known-neuron-function) is open-edit — everyone has edit access. Please help keep it accurate, and separate multiple entries in one cell with `; ` (semicolon + space) for programmatic access.
