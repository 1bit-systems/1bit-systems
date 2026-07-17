# Native tokenizer tool — pure C++17 BPE, zero deps, bit-exact with HF.
# Used by daemon/npu-cppd.py for encode/decode (replaces the broken
# fused-engine --tokenize-only and engine/npu/tokenizer/detokenize paths).
#
#   make            -> builds ./tokenize
#   ./tokenize --model tokenizer.json --encode "Hello, world!"   # 9707 11 1879 0
#   ./tokenize --model tokenizer.json --decode 9707 11 1879 0    # Hello, world!

CXX ?= c++
CXXFLAGS ?= -std=gnu++17 -O2

.PHONY: all clean
all: tokenize

tokenize: tokenize.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f tokenize
