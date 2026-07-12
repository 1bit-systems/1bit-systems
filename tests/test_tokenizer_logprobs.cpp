// test_tokenizer_logprobs.cpp — unit test for tokenizer logprob API (#81)
// Build: g++ -std=c++17 -Iinclude -Iinclude/rocm_cpp -o test_tokenizer_logprobs \
//        tests/test_tokenizer_logprobs.cpp src/tokenizer.cpp -lm
// Run:   ./test_tokenizer_logprobs

#include "rocm_cpp/tokenizer.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>

// ── Create a minimal .htok file in memory for testing ──
// Format: HTOK magic, vocab_size, num_merges, bos_id, eos_id,
//         then vocab entries (uint16_t len + bytes), then merges (a, b, merged_id)
static bool create_test_htok(const char* path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;

    // Magic
    f.write("HTOK", 4);

    // Small test vocabulary (20 tokens) + merges (10 merges)
    uint32_t vocab_size = 20;
    uint32_t num_merges = 10;
    uint32_t bos_id = 1;
    uint32_t eos_id = 2;
    f.write((const char*)&vocab_size, 4);
    f.write((const char*)&num_merges, 4);
    f.write((const char*)&bos_id, 4);
    f.write((const char*)&eos_id, 4);

    // Vocabulary: each token is a GPT-2-mapped byte string.
    // Token 0: "a" (byte 0x61 → codepoint 0x61)
    // Token 1: "b" (BOS)
    // Token 2: "c" (EOS)
    // Token 3: "ab" (merge of 0+1)
    // Token 4: "bc" (merge of 1+2)
    // ... tokens 0-9 are single-byte, 10-19 are multi-byte merges
    const char* vocab[] = {
        "a", "b", "c", "ab", "bc", "abc", "d", "e", "f", "g",
        "ab ", " bc", "abc ", " bc ", "a b", "b c", "ab c", " bc a", "abc d", "e f"
    };

    for (uint32_t i = 0; i < vocab_size; i++) {
        uint16_t len = (uint16_t)strlen(vocab[i]);
        f.write((const char*)&len, 2);
        f.write(vocab[i], len);
    }

    // Merges: (a, b, merged_id, implicit_rank=i)
    // Earlier merges (lower i) should get higher logprobs.
    int merge_triples[][3] = {
        {0, 1, 3},    // rank 0: a + b → ab
        {1, 2, 4},    // rank 1: b + c → bc
        {3, 2, 5},    // rank 2: ab + c → abc
        {5, 0, 10},   // rank 3: abc + ' ' → "abc "
        {0, 10, 11},  // rank 4: ' ' + "abc" → " abc" (wait, this doesn't work with our encoding)
        // Actually let's make simple merges to keep it clean:
        {6, 7, 14},   // rank 5: d + e → "de"
        {14, 8, 15},  // rank 6: de + f → "def"
        {9, 0, 16},   // rank 7: g + a → "ga"
        {3, 4, 17},   // rank 8: ab + bc → "abbc"  
        {5, 6, 18},   // rank 9: abc + d → "abcd"
    };

    for (int i = 0; i < 10; i++) {
        uint32_t a = (uint32_t)merge_triples[i][0];
        uint32_t b = (uint32_t)merge_triples[i][1];
        uint32_t merged = (uint32_t)merge_triples[i][2];
        f.write((const char*)&a, 4);
        f.write((const char*)&b, 4);
        f.write((const char*)&merged, 4);
    }

    return true;
}

// ── Tests ──
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name, expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "FAIL: %s (%s:%d)\n", name, __FILE__, __LINE__); \
        tests_failed++; \
    } else { \
        tests_passed++; \
    } \
} while(0)

int main() {
    // 1. Create test .htok file
    const char* tmp_path = "/tmp/test_tokenizer.htok";
    if (!create_test_htok(tmp_path)) {
        fprintf(stderr, "Failed to create test .htok\n");
        return 1;
    }

    // 2. Load tokenizer
    rcpp_tokenizer_t* tok = nullptr;
    rcpp_status_t rc = rcpp_tokenizer_load(tmp_path, &tok);
    TEST("load succeeds", rc == RCPP_OK && tok != nullptr);

    if (!tok) {
        fprintf(stderr, "Tokenier load returned null\n");
        return 1;
    }

    // 3. Test BOS/EOS IDs
    TEST("BOS ID", rcpp_tokenizer_bos_id(tok) == 1);
    TEST("EOS ID", rcpp_tokenizer_eos_id(tok) == 2);

    // 4. Test logprob ordering: lower-rank merges should have HIGHER logprobs (less negative)
    double lp_ab = rcpp_tokenizer_logprob(tok, 3);  // merge rank 0: a+b→ab
    double lp_bc = rcpp_tokenizer_logprob(tok, 4);  // merge rank 1: b+c→bc
    double lp_abc = rcpp_tokenizer_logprob(tok, 5); // merge rank 2: abc→abc
    double lp_de = rcpp_tokenizer_logprob(tok, 14); // merge rank 5: d+e→de

    printf("  logprobs: ab=%.4f bc=%.4f abc=%.4f de=%.4f\n", lp_ab, lp_bc, lp_abc, lp_de);

    TEST("rank 0 > rank 1", lp_ab > lp_bc);   // ab (rank 0) more likely than bc (rank 1)
    TEST("rank 0 > rank 2", lp_ab > lp_abc);  // ab (rank 0) more likely than abc (rank 2)
    TEST("rank 0 > rank 5", lp_ab > lp_de);   // ab (rank 0) more likely than de (rank 5)
    TEST("rank 1 > rank 5", lp_bc > lp_de);   // bc (rank 1) more likely than de (rank 5)
    TEST("rank 2 > rank 5", lp_abc > lp_de);  // abc (rank 2) more likely than de (rank 5)
    TEST("logprob ≤ 0", lp_ab <= 0.0);         // logprobs are non-positive
    TEST("logprob > -20", lp_ab > -20.0);      // not absurdly low

    // 5. Test unmerged token (BOS=1, not created by any merge)
    double lp_bos = rcpp_tokenizer_logprob(tok, 1);
    printf("  BOS logprob: %.4f\n", lp_bos);
    TEST("BOS logprob is moderate", lp_bos > -15.0 && lp_bos < 0.0);

    // 6. Test invalid token ID
    double lp_invalid = rcpp_tokenizer_logprob(tok, 9999);
    TEST("invalid token gets floor", lp_invalid <= -20.0);

    // 7. Test encode_with_logprobs
    const char* test_text = "abc";
    int ids[32];
    double logprobs[32];
    size_t count = 0;

    rc = rcpp_tokenizer_encode_with_logprobs(tok, test_text, strlen(test_text),
                                              0,  // no BOS
                                              ids, logprobs, 32, &count);
    TEST("encode_with_logprobs succeeds", rc == RCPP_OK);
    TEST("got at least one token", count > 0);

    printf("  encoded \"%s\" → %zu tokens: [", test_text, count);
    for (size_t i = 0; i < count; i++) {
        printf("%d(%.2f)%s", ids[i], logprobs[i], i+1 < count ? ", " : "");
    }
    printf("]\n");

    // 8. Test with BOS token
    rc = rcpp_tokenizer_encode_with_logprobs(tok, test_text, strlen(test_text),
                                              1,  // add BOS
                                              ids, logprobs, 32, &count);
    TEST("encode_with_logprobs with BOS succeeds", rc == RCPP_OK);
    TEST("got at least BOS + 1 token", count >= 2);
    if (count >= 2) {
        TEST("first token is BOS", ids[0] == 1);
    }

    // 9. Test that logprob_for_id is deterministic
    double lp_ab_2 = rcpp_tokenizer_logprob(tok, 3);
    TEST("deterministic", lp_ab == lp_ab_2);

    // 10. Cleanup
    rcpp_tokenizer_free(tok);
    std::remove(tmp_path);

    // ── Results ──
    printf("\n=== Results: %d passed, %d failed ===\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
