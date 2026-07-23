#pragma once
// dataset.h — JSON dataset loader + tokenizer for LoRA training
// Pure C++23, no Python, no HuggingFace datasets.
// Reads JSONL format: {"instruction": "...", "output": "..."}

#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <algorithm>

// ── Simple JSONL dataset reader ───────────────────────────────────────────
// Format: one JSON object per line
// Fields: instruction, input, output (or question, answer)
// Returns tokenized sequences (vocab IDs)

struct TrainExample {
    std::vector<int> input_ids;
    std::vector<int> label_ids;
    int n_tokens;
};

struct Dataset {
    std::vector<TrainExample> examples;
    int vocab_size = 0;
    int max_seq_len = 0;
    
    // Load from JSONL file with format {"instruction": "...", "output": "..."}
    // Also supports: {"question": "...", "answer": "..."} or {"text": "..."}
    bool load(const char* path, int max_len = 512, bool verbose = true) {
        FILE* f = fopen(path, "rb");
        if (!f) { perror("fopen"); return false; }
        
        fseek(f, 0, SEEK_END);
        long fsize = ftell(f);
        fseek(f, 0, SEEK_SET);
        std::string data((size_t)fsize, '\0');
        fread(data.data(), 1, fsize, f);
        fclose(f);
        
        // Simple JSONL parser — finds lines and extracts text
        size_t pos = 0;
        int count = 0;
        while (pos < data.size()) {
            // Find line boundary
            size_t eol = data.find('\n', pos);
            if (eol == std::string::npos) eol = data.size();
            std::string line = data.substr(pos, eol - pos);
            pos = eol + 1;
            
            if (line.size() < 10) continue;
            
            // Extract instruction and output from JSON
            auto extract = [&](const std::string& key) -> std::string {
                auto kpos = line.find("\"" + key + "\"");
                if (kpos == std::string::npos) return "";
                kpos = line.find(':', kpos + key.size() + 2);
                if (kpos == std::string::npos) return "";
                kpos = line.find('"', kpos);
                if (kpos == std::string::npos) return "";
                kpos++; // skip opening quote
                auto end = kpos;
                while (end < line.size() && line[end] != '"') {
                    if (line[end] == '\\') end++; // skip escape
                    end++;
                }
                return line.substr(kpos, end - kpos);
            };
            
            std::string instr = extract("instruction");
            if (instr.empty()) instr = extract("question");
            std::string inp = extract("input");
            std::string out = extract("output");
            if (out.empty()) out = extract("answer");
            std::string plain = extract("text");
            
            if (out.empty() && !plain.empty()) {
                // Self-supervised: text is both input and label
                inp = plain;
                out = "";
            }
            
            if (instr.empty() && out.empty()) continue;
            
            // Format as instruction:response pair
            std::string text;
            if (!inp.empty())
                text = "### Instruction:\n" + instr + "\n\n### Input:\n" + inp + "\n\n### Response:\n" + out;
            else if (!instr.empty())
                text = "### Instruction:\n" + instr + "\n\n### Response:\n" + out;
            else
                continue;
            
            if (text.size() < 10) continue;
            
            // Simple ASCII tokenization (byte-level, maps to 0-255 range)
            // In production, replace with the actual tokenizer (engine has one)
            std::vector<int> ids;
            ids.push_back(1); // BOS
            for (unsigned char c : text) {
                if (c == '\0') continue;
                ids.push_back((int)c + 2); // shift by 2 to avoid special tokens
            }
            ids.push_back(2); // EOS
            
            if ((int)ids.size() > max_len) continue;
            if ((int)ids.size() < 5) continue;
            
            // Labels = input_ids (causal LM)
            TrainExample ex;
            ex.input_ids = ids;
            ex.label_ids = ids;
            ex.n_tokens = (int)ids.size();
            examples.push_back(std::move(ex));
            count++;
        }
        
        max_seq_len = max_len;
        if (verbose)
            printf("  Loaded %d examples from %s\n", count, path);
        return count > 0;
    }
    
    // Shuffle and take a subset
    void shuffle_and_slice(int seed, float pct = 100.0f) {
        std::mt19937 rng(seed);
        std::shuffle(examples.begin(), examples.end(), rng);
        int n = std::max(1, (int)(examples.size() * pct / 100.0f));
        examples.resize(n);
    }
    
    // Get a batch
    struct Batch {
        std::vector<int> input_ids;  // [batch, seq_len]
        std::vector<int> label_ids;  // [batch, seq_len]
        int batch_size, seq_len;
    };
    
    Batch get_batch(int start, int bs) const {
        int n = std::min(bs, (int)examples.size() - start);
        Batch b;
        b.batch_size = n;
        b.seq_len = max_seq_len;
        b.input_ids.resize((size_t)n * max_seq_len, 0);
        b.label_ids.resize((size_t)n * max_seq_len, -1);
        
        for (int i = 0; i < n; i++) {
            const auto& ex = examples[start + i];
            for (int j = 0; j < ex.n_tokens && j < max_seq_len; j++) {
                b.input_ids[(size_t)i * max_seq_len + j] = ex.input_ids[j];
                b.label_ids[(size_t)i * max_seq_len + j] = ex.label_ids[j];
            }
        }
        return b;
    }
};
