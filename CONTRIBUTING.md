# Contributing to Moonshine

We appreciate you contributing to Moonshine! We are using Github Issues to track both
bugfixes and feature requests.

Please reach out on
[Slack](https://join.slack.com/t/moonshinecommunity/shared_invite/zt-1rg1vnvmt-pleUR7TducaDiAhcmnqAQQ)
if you have questions, and join #contributing.

## Setup

To setup the environment, simply clone the repo and create a virtual environment, then
run:

```
pip install -e .
```

## Contributing

To make a submission to Moonshine:

1. Fork a copy of the Moonshine library to your own account.
1. Clone your fork locally and add the Moonshine repo as a remote:

```
git clone git@github.com:{YOUR USERNAME}/streaming.git
cd moonshine
git remote add upstream https://github.com/moonshinelabs-ai/moonshine.git
```

3. Create a branch and propose your changes:

```
git checkout -b username/feature-name
```

4. We do not have automatic pre-commit formatting, but we do use Github actions to check
   linting and types. Please run `black .`, `isort .` and `mypy .` before submitting
   your code for review. Once your code is passing the Github action CI, submit a pull
   request!

## Documentation

To update the documentation, you will need to install the documentation requirements in
`docs_requirements.txt` to a virtual environment. Then, from the `docs/` directory, run
`make html` to build the docs locally.
