install:
	pip install -U pip &&\
	pip install -r requirements.txt

format:
	python -m black src/*.py *.py