install:
	pip install -U pip &&\
	pip install -r requirements.txt

format:
	black *.py src/*.py tests/*.py

lint:
	pylint --disable=R,C *.py src/*.py

test:
	python -m pytest tests

refactor: format lint

all: install format lint test