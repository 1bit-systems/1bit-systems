FROM ubuntu:26.04

RUN apt-get update && apt-get install -y \
    g++ libxrt2 libxrt-npu2 libxrt-dev \
    && rm -rf /var/lib/apt/lists/*

COPY . /opt/1bit-systems
WORKDIR /opt/1bit-systems

RUN gcc -c -std=c11 -O3 -o engine/npu/build/dequant_q4nx.o engine/npu/src/dequant_q4nx.c && \
    g++ -std=c++23 -O3 -march=native -fopenmp -ffast-math -o /usr/local/bin/1bit-npu \
      engine/npu/src/npu_engine_all.cpp engine/npu/build/dequant_q4nx.o \
      -Iengine/npu/src -lxrt_coreutil -luuid -lm -ldl && \
    g++ -std=c++23 -O3 -o /usr/local/bin/1bit-server \
      packaging/binary/server.cpp

EXPOSE 8081

ENTRYPOINT ["1bit-server"]
CMD ["8081"]
