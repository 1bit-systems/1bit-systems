// tokenizer.h — Gemma/Qwen-style BPE tokenizer
#pragma once
#include <string>
#include <vector>
#include <unordered_map>

struct BPETokenizer {
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int> token_to_id;
    int vocab_size = 0;
    
    // Special token IDs (Gemma defaults)
    int pad_id = 0;
    int eos_id = 1;
    int bos_id = 2;
    int unk_id = 3;

    bool load(const std::string& path);
    std::vector<int> encode(const std::string& text);
    std::string decode(const std::vector<int>& tokens);
};
