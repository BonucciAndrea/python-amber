# python-amber -- development shortcuts. The real build is setup.py; this is
# just the four commands you actually type.
AMBER_SRC ?= ../amber

.PHONY: help dev test wheel sdist manylinux clean lint

help:
	@echo "make dev        editable install against \$$AMBER_SRC ($(AMBER_SRC))"
	@echo "make test       run the pytest suite"
	@echo "make wheel      build a self-contained wheel"
	@echo "make manylinux  build + auditwheel repair (run inside a manylinux image)"
	@echo "make sdist      build a source distribution"
	@echo "make clean      remove build artefacts and the bundled engine"

dev:
	AMBER_SRC=$(AMBER_SRC) python3 -m pip install -e ".[dev]"

test:
	python3 -m pytest tests -q

wheel:
	AMBER_SRC=$(AMBER_SRC) scripts/build_wheel.sh

manylinux:
	AMBER_SRC=$(AMBER_SRC) scripts/build_wheel.sh --manylinux

sdist:
	AMBER_SRC=$(AMBER_SRC) scripts/build_wheel.sh --sdist

clean:
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache
	rm -rf src/amber/__pycache__ src/amber/libamber.so src/amber/libamber.dylib src/amber/kmodules
	find . -name '*.so' -path '*/amber/*' -delete
