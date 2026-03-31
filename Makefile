PYTHONPATH=src

.PHONY: install test demo report benchmark pages hf-example

install:
	python3 -m pip install -e .[dev]

test:
	PYTHONPATH=$(PYTHONPATH) pytest

demo:
	PYTHONPATH=$(PYTHONPATH) python3 demos/reference_demo.py

report:
	PYTHONPATH=$(PYTHONPATH) python3 demos/visual_report.py

benchmark:
	PYTHONPATH=$(PYTHONPATH) python3 benchmarks/synthetic_distortion.py

pages:
	PYTHONPATH=$(PYTHONPATH) python3 scripts/build_pages.py

hf-example:
	PYTHONPATH=$(PYTHONPATH) python3 examples/transformers_generate.py --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0
