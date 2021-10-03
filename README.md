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
- [Usage](#usage)
- [Website](#website)

## Development
It is good practice to develop in a virtual environment. Note, this jupyter book was written using `python 3.8` and on the `Ubuntu` (Linux) operating system (OS). As such, all commands are installed for this setup and may not work for other OS's. To create a virtual environment called `venv`, execute:
```bash
python3 -m venv venv
```
To activate it, execute
```bash
source venv/bin/activate
```

- Execute `pip install -r requirements.txt` to install requirements for development. Please see `requirements.txt` for development requirements.

## Usage
To build the jupyter book as HTML, please execute `jupyter-book build --all book/`. To build a PDF version of the book, please run `jupyter-book build book/ --builder pdfhtml`. **Note**, you will have to build a html version first. For further information, please [click here](https://jupyterbook.org/advanced/pdf.html?highlight=build%20pdf). The PDF file can be found in the [docs](/docs) folder.

## Website
The content of this jupyter book is [hosted here](https://reidya3.github.io/CA4015_First_Assignment/Introduction.html). Github actions are utilized to automatically build the book and update the website when a `push` or `pull request` event occurs on the main branch.