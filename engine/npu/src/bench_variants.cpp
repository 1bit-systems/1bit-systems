/** Variant benchmark — patch into npu_engine_i8.cpp main() to test prompts/decode */
// Insert at line ~129 instead of fixed pt[]={...}
// This file documents the benchmark variants used. Copy sections into npu_engine_i8.cpp

// === BENCH 1: Prompt-length scaling ===
int pt_1[]={151643};                         int npt=1;   // BOS only
int pt_4[]={151643,872,198,11852};           int npt=4;   // short
int pt_9[]={151643,872,198,11852,151644,198,151643,77091,198}; int npt=9;  // chat
int pt_20[]={151643,872,198,11852,151644,198,151643,77091,198,  // extended
             151643,872,198,11852,151644,198,151643,77091,198,151643,872};

// === BENCH 2: Decode-length scaling ===
// int ng=N;  // 4, 8, 16, 32, 64 tokens

// === BENCH 3: Multiple runs for consistency ===
// Run 3× and check token diversity
