.PHONY: validate inherit metrics marts report upload

validate:
	PYTHONPATH=src python -m ddm.pipeline validate

inherit:
	PYTHONPATH=src python -m ddm.pipeline inherit

metrics:
	PYTHONPATH=src python -m ddm.pipeline metrics

marts:
	PYTHONPATH=src python -m ddm.pipeline marts

report:
	PYTHONPATH=src python -m ddm.pipeline report

upload:
	PYTHONPATH=src python scripts/upload_to_db.py

upload-fresh:
	PYTHONPATH=src python scripts/upload_to_db.py --fresh
