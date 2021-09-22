all:
	$(error please pick a target)

.PHONY: help setup-dev build-html build-html clean lint ci

help:
	@echo "---------------HELP-----------------"
	@echo "To install dev requirements, type make setup-dev"
	@echo "To test the project, type make test"
	@echo "To remove build artiacts, type make clean"
	@echo "To perform linting, type make lint"
	@echo "To manually run the pre-commit hooks, type make ci"
	@echo "------------------------------------"

setup-dev: ## Setup development environment
	pip install -r requirements.txt

build-html:
	jupyter-book build --all book/

build-pdf:
	jupyter-book build book/ --builder pdfhtml

clean:
	jupyter-book clean book/

lint:## Disabled duplicate code warning as similar but slightly different expected outputs in various tests triggered this warning
	PYTHONPATH=src/:$(PYTHONPATH) pylint src/* test/* --fail-under=8.5 --max-line-length=120 -d duplicate-code

ci:
	pre-commit run --hook-stage manual
