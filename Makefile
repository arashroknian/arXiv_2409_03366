PY=python3.9
# PY=py3.10

BIN=venv/bin

default: venv

venv:
	$(PY) -m venv venv
	$(BIN)/pip install -r requirements.txt
	git clone https://github.com/pmgbergen/porepy
	cd porepy; \
		git checkout a46c9652bbc955b259ebcdacfdf846c183910dcd; \
		git apply --ignore-space-change ../porepy.patch;
	$(BIN)/pip install -e porepy
	$(BIN)/pip install -e .

test:
	$(BIN)/python notebooks/esempio.py

nb:
	$(BIN)/python -m notebook

clean:
	rm -rf venv
	rm -rf porepy
	rm -rf ddf.egg-info
