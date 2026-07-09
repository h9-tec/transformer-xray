PY := model/.venv/bin/python

.PHONY: data train sample export golden artifacts dev build test bench

data:
	$(PY) model/data.py

train: data
	$(PY) model/train.py

sample:
	$(PY) model/sample.py

export:
	$(PY) model/export.py
	mkdir -p web/public/model
	cp model/artifacts/weights.bin model/artifacts/config.json model/artifacts/tokenizer.json web/public/model/

golden:
	$(PY) model/golden.py

artifacts: export golden

dev:
	cd web && npx pnpm dev

build:
	cd web && npx pnpm build

test:
	cd web && npx pnpm test

bench:
	$(MAKE) -C kernels bench
