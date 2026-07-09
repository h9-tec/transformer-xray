# Teaching kernels — compile + verify + time on a real NVIDIA GPU.
#
#   make bench            # build & run, prints JSON
#   make bench ARCH=compute_86   # override target arch (default: PTX for wide JIT)
#
# PTX-only output (-arch=$(ARCH) -code=$(ARCH)) lets newer GPUs (e.g. Blackwell
# with an older toolkit) JIT the kernels through the driver.

ARCH ?= compute_86
NVCC ?= nvcc
SRC  := 01_embedding.cu 02_layernorm.cu 03_matmul.cu 04_attention.cu 05_elementwise.cu bench_main.cu

build/bench: $(SRC) common.cuh
	@mkdir -p build
	$(NVCC) -O3 -arch=$(ARCH) -code=$(ARCH) $(SRC) -o $@

.PHONY: bench clean
bench: build/bench
	./build/bench

clean:
	rm -rf build
