# Python Guidelines and Coding style

Template code and examples for python project respecting the [PEP8 coding style](https://peps.python.org/pep-0008/).

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) ![linter](https://github.com/pollen-robotics/python-coding-guidelines/actions/workflows/lint.yml/badge.svg) ![pytest](https://github.com/pollen-robotics/python-coding-guidelines/actions/workflows/pytest.yml/badge.svg) ![coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/FabienDanieau/58642e8fe4589e710e26627e39ff92d7/raw/covbadge.json)


## Quick start

List of steps to follow to adapt the template to your project.

1. Make your repo with "Use this template" button
2. Clone the new repo
3. [Recommended] Make a virtual environment
4. [Optional] Set up [git lfs](#git) and *.gitattributes*
4. Activate pre-commit hook
```console
mv scripts/git_hooks/pre-commit .git/hooks/
```
5. Install dev tools
```console
pip install -e .[dev]
```
6. [Configure](#ide---linting) Visual Code with the linters
7. [Adapt](#unit-tests-and-test-coverage) workflow and bagdes links to this repo


## Configuration

## Installation

Dependencies are detailed in the `setup.cfg` file. To install this package locally, run:
```
pip install -e .[dev]
```
Include *[dev]* for optional development tools.

The dependencies are listed in the ```setup.cfg``` file and will be installed if you install this package locally with:
```
pip install -e .[dev]
```
use *[dev]* for optional development tools.


Once this is done, you should be able to import the Python package anywhere on your system with:
```console
import example
```

An exemple executable is also installed
```console
example_entry_point
```


### git

A *.gitignore* file ensures that the Python temporary files are not committed. It is adapted from [here](https://github.com/github/gitignore/blob/main/Python.gitignore).

[git LFS](https://git-lfs.com/) can be configured to manage all non-script files (3D models, deep learning models, images, etc.). The list of files is defined in *gitattributes_example*. If you wish to utilize LFS, rename this file to *.gitattributes* and then run ```git lfs install```. It's also a good practice to maintain another repository as a submodule containing all the data.

A git hook can be set up to automatically check for PEP8 compliance before committing. Refer to *scripts/git_hooks*. Installing this is recommended as the GitHub workflows will perform the same checks.


### IDE - Linting

Visual Studio Code is the recommended IDE. Ensure you have the Python extension installed and [configure](https://dev.to/adamlombard/how-to-use-the-black-python-code-formatter-in-vscode-3lo0) VS Code to auto-format your code using [black](https://black.readthedocs.io).

[isort](https://pycqa.github.io/isort/) verifies the order of Python imports. [Set up](https://github.com/microsoft/vscode-isort#import-sorting-on-save) VS Code to handle this automatically.

[Flake8](https://flake8.pycqa.org) will conduct additional checks. Choose **flake8** as the [linter for VS Code](https://code.visualstudio.com/docs/python/linting). Errors will be directly highlighted within the code.

Lastly, [mypy](https://mypy.readthedocs.io/en/stable/index.html) will perform static type checking. In this guide, it's set to run in **strict** mode. Feel free to [adjust the constraints](https://mypy.readthedocs.io/en/stable/getting_started.html?highlight=strict#strict-mode-and-configuration) if they don't align with your project's needs.

These tools are configured in the **setup.cfg** file. Their versions are predefined to prevent discrepancies between local and remote checks. It's advisable to keep them updated.


**A code not compliant with PEP8 guidelines will not be merged.**

## Workflow

### Branches

The git workflow is based on [gitflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow):

![git workflow](https://wac-cdn.atlassian.com/dam/jcr:34c86360-8dea-4be4-92f7-6597d4d5bfae/02%20Feature%20branches.svg?cdnVersion=805)

The default branch is **develop**, which serves as the working branch. **main** is reserved solely for stable code (releases). Both of these branches are protected, meaning direct pushes are disallowed. All new development **MUST** commence in a feature branch, derived from develop, and then be merged back into develop via a pull request. Once the code is deemed stable, the develop branch can also be merged into main through another pull request.


### Issues

The issue system is a great tool to track problem, bugs, new features, etc. Commits and branches can be linked to a feature. Thus, before any development, a new issue **MUST** be created.

### Procedure

1. Create an [issue](https://github.com/pollen-robotics/unity-workflow/issues) with the *New Issue* button. Document the problem as much as possible (images, url, ...)
2. Create a new branch. The easiest way is to do it directly from the [issue](https://github.blog/changelog/2022-03-02-create-a-branch-for-an-issue/). Thus, all branch names will be consistent, and the PR will directly close the issue.
3. Checkout your branch locally:
 ```
 git fetch origin
 git checkout <branch-name>
 ```
4. Work in the local branch and commit often. Each commit must be formatted as
 ```
<tag> #<issue number> : <message>
 ```
 With *tag* being fix, bug or any label from the issue system, *issue number* the number of the issue, and *message* the mandatory message associated to the commit. Issue number is important so github can link the commit to the issue.
 
 5. Create unit test to validate the new code (see below).

 6. When the work is implemented, create a *pull request*. Easiest way is to do it from the branch page of the repo.
 
 7. At the PR creation, unit tests will be computed and result reported. The branch will not be merged until they pass. Besides, the project is configured so a developer cannot merge his/her own code. An external review is mandatory.
 8. Merge is completed, go to step 1.

 ### Unit tests and test coverage

 Unit tests must be written to ensure code robustness. [Pytest](https://docs.pytest.org) is the recommended tool. Examples are provided in the *tests* folder.
 
It is recommended to have at least 90% of the code tested. The [coverage](https://coverage.readthedocs.io) package provide this metric.

 The developer must run the test locally before committing any new code. Make sure that *pytest* and *coverage* are installed and run at the root level:
 ```
 coverage run -m pytest
 ```
Then, if all tests are sucessful:
 ```
 coverage report
 ```
 These tests are automatically performed by a github action when a pull request is created.

 _Note that when creating a new repo from this template, you will need to first configure the Make badge action in the [pytest.yml](https://github.com/pollen-robotics/python-template/blob/develop/.github/workflows/pytest.yml#L42-L53) file. Follow [those instructions](https://github.com/schneegans/dynamic-badges-action/tree/v1.6.0/#configuration) if you don't have a gist secret and id yet._ 

 ## Coding style

 The main guidelines for the coding style is defined by [PEP8](https://peps.python.org/pep-0008/). You can directly refer to the examples in the code.

 Specific choices are detailed in dedicated document:
 - for the mathematical notation please refer to [Coding convention for Maths](docs/convention_maths.md)