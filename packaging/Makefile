# 1bit.systems — build & package (v2026.07.02-all5models)
# One binary. 5 models. 120KB. Zero deps.
.PHONY: all npu gpu package-deb package-aur package-brew clean

all: npu

npu:
	gcc -c -std=c11 -O3 -o engine/npu/build/dequant_q4nx.o engine/npu/src/dequant_q4nx.c
	g++ -std=c++23 -O3 -march=native -fopenmp -ffast-math \
		-o engine/npu/build/npu_engine_all \
		engine/npu/src/npu_engine_all.cpp \
		engine/npu/build/dequant_q4nx.o \
		-Iengine/npu/src \
		-I$(XRT)/include -L$(XRT)/lib64 \
		-lxrt_coreutil -luuid -lm -ldl

gpu:
	cd /home/bcloud/zinc && zig build -Doptimize=ReleaseFast

package-deb: npu
	mkdir -p packaging/deb/usr/bin packaging/deb/usr/lib/1bit
	cp engine/npu/build/npu_engine_all packaging/deb/usr/bin/1bit-npu
	cp engine/npu/build/dequant_q4nx.o packaging/deb/usr/lib/1bit/
	cp packaging/binary/1bit-server packaging/deb/usr/bin/1bit-server
	dpkg-deb --build packaging/deb 1bit-systems.deb

package-snap: npu
	cp engine/npu/build/npu_engine_all .
	cp packaging/binary/1bit-server .
	cp engine/npu/build/dequant_q4nx.o .
	snapcraft

package-aur:
	cp packaging/aur/PKGBUILD .
	makepkg -si

package-brew:
	cp packaging/homebrew/1bit-systems.rb /opt/homebrew/Library/Taps/homebrew/homebrew-core/Formula/

package-docker:
	docker build -t 1bit-systems -f packaging/docker/Dockerfile .

package-tarball: npu
	mkdir -p 1bit-systems-$(shell date +%Y.%m.%d)/bin 1bit-systems-$(shell date +%Y.%m.%d)/lib
	cp engine/npu/build/npu_engine_all 1bit-systems-$(shell date +%Y.%m.%d)/bin/1bit-npu
	cp engine/npu/build/dequant_q4nx.o 1bit-systems-$(shell date +%Y.%m.%d)/lib/
	tar czf 1bit-systems-$(shell date +%Y.%m.%d)-linux-amd64.tar.gz 1bit-systems-$(shell date +%Y.%m.%d)/

clean:
	rm -f engine/npu/build/npu_engine_all engine/npu/build/dequant_q4nx.o
