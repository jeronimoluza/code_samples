.PHONY = create ipyk clean install-reqs install-pyhdf install-basemap install

SHELL := /bin/bash

REPO=$(shell basename $(CURDIR))


create:
	conda create --name $(REPO) -y; \

ipyk:
	python -m ipykernel install --user --name=$(REPO);

clean:
	conda env remove --name $(REPO) -y;

install-reqs:
	conda install pip -y; \
	python -m pip install -r requirements.txt; \

install-pyhdf:
	conda install -c conda-forge pyhdf -y;

install-basemap:
	conda install -c conda-forge basemap -y;

install:
	install-reqs install-pyhdf install-basemap