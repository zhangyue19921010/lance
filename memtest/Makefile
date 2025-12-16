.PHONY: build test lint format clean

build:
	cargo build
	cp target/debug/libmemtest.so python/memtest/
	pip install -e .

build-release:
	cargo build --release
	cp target/release/libmemtest.so python/memtest/
	pip install -e .

test:
	LD_PRELOAD=./python/memtest/libmemtest.so pytest python/tests/ -v

lint:
	cargo clippy -- -D warnings
	ruff check python/

format:
	cargo fmt
	ruff format python/

clean:
	cargo clean
	rm -rf target/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.so" -delete
