#pragma once
#include "onebp_format.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>

struct OnebpTensor {
    std::string name;
    int ndim;
    std::vector<uint32_t> dims;
    uint64_t offset;
    uint64_t bytes;
};

struct OnebpModel {
    OnebpHeader header;
    std::vector<OnebpTensor> tensors;
    uint8_t* data = nullptr;
    size_t file_size = 0;
    uint64_t data_section_offset = 0;  // byte offset from data start to weight data
    int fd = -1;

    ~OnebpModel();
    bool load(const char* path);
    uint8_t* tensor_data(const OnebpTensor& t);
};
