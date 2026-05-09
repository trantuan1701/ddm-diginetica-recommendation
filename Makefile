.PHONY: validate inherit metrics marts report

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
