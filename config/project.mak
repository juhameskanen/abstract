# Makefile for building masterpiece projects.

PYPROJECT = pyproject.toml

PROJECT_NAME := $(shell sed -n 's/^name = "\(.*\)"/\1/p' $(PYPROJECT))
PROJECT_VERSION := $(shell sed -n 's/^version = "\(.*\)"/\1/p' $(PYPROJECT))



help:
	@echo "make clean package upload install unittest pyright coverage mypy html unitcov"

version:
	@echo "$(PROJECT_NAME): $(PROJECT_VERSION)"

package:
	@python3 -m build

install:
	@python3 -m pip install -e .

uninstall:
	@python3 -m pip uninstall $(PROJECT)

check:
	@python3 -m twine check dist/*

upload:
	@python3 -m twine upload --repository pypi dist/*

clean:
	@rm -r -f dist public .coverage *.log.* *.json

unittest:
	@pwd
	@python3 -m unittest discover

pyright:
	pyright

coverage:
	pytest --cov --cov-report=xml

unitcov:
	pytest --cov

mypy:
	mypy --show-traceback .

html:
	$(MAKE) -C docs $@ $(MAKEOPTS) || exit $$?

commit:
	git commit . -m "$(MESSAGE)"

push:
	git push

pull:
	git pull
