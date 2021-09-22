<h1 align="center">
  <img alt="Iowa Gambling logo" src="./book/images/IGT-logo.png" height="115px" />
  <br/>
  Iowa Gambling Clustering Analysis (CA4015)
</h1>
<h3 align="center">
  Anthony Reidy, 18369643.
  <br/><br/><br/>
</h3>

## Table of Contents
- [Development](#development)
  - [Setup](#setup)
  - [Code Quaility](#pre-commit-hooks---pylint--black) 
- [Usage](#usage)

## Development
### Setup 
It is good practice to develop in a virtual environment(VE). Note, this jupyter book was written using `python 3.7` and on the `Window` operating system (OS). As such, all commands are installed for this system annd may not work for other OS's. To create a VE called `venv`, execute:
```bash
python -m venv venv
```
To activate it, execute
```bash
source venv/bin/activate
```

- Execute `make setup-dev` to install requirements for development. 

- To see all make commands, run `make help`

Please see `requirements.txt` for development requirements.

### Pre-commit hooks - Pylint & Black
Git hook scripts are useful for identifying code quaility issues before submission to code review. This repository contains the following pre-commit hooks:
- **Pylint**: A checker that provides warnings/info on the coding standard of a repository.  Example warnings include line length, incorrect sequence of imports, whitespace etc.
- **Black**: An uncompromising Python code formatter. It applies changes and modifies the layout of files.

Run `pre-commit install` to set up the git hook scripts. Now, pre-commit will run automatically on a git commit! If a hook fails, it will abort the commit and in the case of Black, it will change the files in place. The configuration of pre-commit can be found in [.pre-commit-config.yaml](.pre-commit-config.yaml). Note, the hooks are only applied to files under the [book](book) directory, respectively. To manually run the hooks, execute `make ci`.

## Usage
- To build the jupter book as HTML, please execute make build-html.

- To build a pdf verision of the book, please run make build-pdf. **Note**, you will have to build a html verision first. For futher information please [click here](https://jupyterbook.org/advanced/pdf.html?highlight=build%20pdf)

**Insert jupyter book commands later (Makefile)**
