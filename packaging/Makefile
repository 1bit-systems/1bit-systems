# 1bit.systems — build & package
.PHONY: all npu gpu package-deb package-aur package-brew clean

all: npu

npu:
	gcc -c -O3 -o engine/npu/build/dequant_q4nx.o engine/npu/src/dequant_q4nx.c
	g++ -std=c++23 -O3 -o engine/npu/build/npu_engine \
		engine/npu/src/npu_engine_i8.cpp \
		engine/npu/build/dequant_q4nx.o \
		-I$(XRT)/include -L$(XRT)/lib64 \
		-lxrt_coreutil -luuid -lm -ldl

gpu:
	cd /home/bcloud/zinc && zig build -Doptimize=ReleaseFast

package-deb: npu
	mkdir -p packaging/deb/usr/bin packaging/deb/usr/lib/1bit
	cp engine/npu/build/npu_engine packaging/deb/usr/bin/1bit-npu
	cp engine/npu/build/dequant_q4nx.o packaging/deb/usr/lib/1bit/
	dpkg-deb --build packaging/deb 1bit-systems.deb

package-aur:
	cp packaging/aur/PKGBUILD .
	makepkg -si

package-brew:
	cp packaging/homebrew/1bit-systems.rb /opt/homebrew/Library/Taps/homebrew/homebrew-core/Formula/

clean:
	rm -f engine/npu/build/npu_engine engine/npu/build/dequant_q4nx.o engine/gpu/zig-out/bin/zinc
