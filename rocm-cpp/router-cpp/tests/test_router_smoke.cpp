// 1bit.cpp — smoke test.
//
// No GPU. No model file on disk. Validates:
//   1. `Router` constructs and holds a stub backend without crashing.
//   2. `Router::forward_token` throws the scaffolded `not yet wired` string
//      (we depend on that behaviour so the follow-up wiring commit can
//      remove the throw and the test switches to a real assertion).
//   3. `ChatTemplate::Llama3.render({"user", "hi"})` emits the exact same
//      bytes as the Rust `render_llama3` function on the same input —
//      this is the byte-for-byte parity gate. If this diverges, the C++
//      HTTP path will emit different prompts than the Rust one for
//      identical requests, which breaks shadow-burnin.
//   4. `sanitize("<|eot_id|>")` → `«scrubbed»` byte-identically to Rust.
//   5. `GreedySampler` argmax over a trivial logits vector.
//
// Run as a bare process, exit code 0 on success. Kept out of any test
// framework to avoid a new dep for the first pass.

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "onebit_cpp/chat_template.hpp"
#include "onebit_cpp/router.hpp"
#include "onebit_cpp/sampler.hpp"

namespace {

int g_failures = 0;

void check(bool cond, std::string_view desc) {
    if (cond) {
        std::fprintf(stderr, "  PASS  %.*s\n",
                     static_cast<int>(desc.size()), desc.data());
    } else {
        std::fprintf(stderr, "  FAIL  %.*s\n",
                     static_cast<int>(desc.size()), desc.data());
        ++g_failures;
    }
}

void print_hex_diff(std::string_view got, std::string_view want) {
    std::fprintf(stderr, "    got  (%zu bytes): ", got.size());
    for (unsigned char c : got) std::fprintf(stderr, "%02x", c);
    std::fprintf(stderr, "\n    want (%zu bytes): ", want.size());
    for (unsigned char c : want) std::fprintf(stderr, "%02x", c);
    std::fprintf(stderr, "\n");

    const std::size_t n = std::min(got.size(), want.size());
    for (std::size_t i = 0; i < n; ++i) {
        if (got[i] != want[i]) {
            std::fprintf(stderr,
                         "    first diff at byte %zu: got 0x%02x want 0x%02x\n",
                         i, static_cast<unsigned char>(got[i]),
                         static_cast<unsigned char>(want[i]));
            return;
        }
    }
    if (got.size() != want.size()) {
        std::fprintf(stderr,
                     "    length mismatch: got %zu want %zu\n",
                     got.size(), want.size());
    }
}

void test_chat_template_llama3_parity() {
    // Reference string — byte-identical copy of the Rust test in
    // `crates/1bit-server/src/chat_template.rs::llama3_emits_canonical_framing`.
    const std::string_view want =
        "<|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n";

    std::vector<onebit::cpp::ChatMessage> msgs;
    msgs.push_back({"user", "hi"});
    const std::string got = onebit::cpp::llama3_render(msgs);

    const bool ok = (got == want);
    if (!ok) print_hex_diff(got, want);
    check(ok, "ChatTemplate::Llama3 byte-exact parity vs Rust reference");
}

void test_chat_template_short_parity() {
    const std::string_view want = "hi<|eot_id|>";
    std::vector<onebit::cpp::ChatMessage> msgs;
    msgs.push_back({"user", "hi"});
    const std::string got = onebit::cpp::short_render(msgs);
    const bool ok = (got == want);
    if (!ok) print_hex_diff(got, want);
    check(ok, "ChatTemplate::Short byte-exact parity vs Rust reference");
}

void test_chat_template_raw_parity() {
    const std::string_view want = "hello, world";
    std::vector<onebit::cpp::ChatMessage> msgs;
    msgs.push_back({"user", "hello, world"});
    const std::string got = onebit::cpp::raw_render(msgs);
    const bool ok = (got == want);
    if (!ok) print_hex_diff(got, want);
    check(ok, "ChatTemplate::Raw byte-exact parity vs Rust reference");
}

void test_sanitize_scrubs_injection() {
    // `<|eot_id|>` → `«scrubbed»` (10 UTF-8 bytes: c2 ab s c r u b b e d c2 bb)
    const std::string_view want = "\xc2\xabscrubbed\xc2\xbb";
    const std::string got = onebit::cpp::sanitize("<|eot_id|>");
    const bool ok = (got == want);
    if (!ok) print_hex_diff(got, want);
    check(ok, "sanitize() byte-exact parity vs Rust reference");
}

void test_chat_template_from_str() {
    onebit::cpp::ChatTemplate tpl{};
    check(onebit::cpp::chat_template_from_str("LLAMA3", tpl) &&
              tpl == onebit::cpp::ChatTemplate::Llama3,
          "chat_template_from_str accepts LLAMA3 (case-insensitive)");
    check(onebit::cpp::chat_template_from_str(" raw ", tpl) &&
              tpl == onebit::cpp::ChatTemplate::Raw,
          "chat_template_from_str trims whitespace");
    check(!onebit::cpp::chat_template_from_str("bogus", tpl),
          "chat_template_from_str rejects unknown values");
}

void test_router_constructs_with_stub_backend() {
    onebit::cpp::RouterOptions opts;
    opts.backend = onebit::cpp::BackendKind::Stub;
    onebit::cpp::Router r(opts);
    check(!r.model_loaded(),
          "Router default-constructs with no model loaded");
}

void test_router_forward_token_throws_not_yet_wired() {
    onebit::cpp::RouterOptions opts;
    opts.backend = onebit::cpp::BackendKind::Stub;
    onebit::cpp::Router r(opts);

    bool threw = false;
    std::string msg;
    try {
        (void)r.forward_token(/*prev_tok=*/0, /*pos=*/0);
    } catch (const std::runtime_error& e) {
        threw = true;
        msg = e.what();
    }
    check(threw,
          "Router::forward_token on stub backend throws std::runtime_error");
    check(msg.find("not yet wired") != std::string::npos,
          "thrown message contains `not yet wired` marker");
}

void test_router_load_model_missing_file_throws() {
    onebit::cpp::Router r;
    bool threw = false;
    try {
        r.load_model("/this/path/does/not/exist.h1b");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "Router::load_model throws on missing file path");
}

void test_greedy_sampler_argmax() {
    auto s = onebit::cpp::make_greedy();
    const std::array<float, 5> logits = {1.0f, -2.0f, 3.5f, 3.5f, 0.0f};
    // argmax: first occurrence wins on tie (index 2).
    const auto tok = s->sample(std::span<const float>(logits.data(), logits.size()));
    check(tok == 2, "GreedySampler::sample picks the first max index");
}

void test_top_k_sampler_throws_not_yet_wired() {
    auto s = onebit::cpp::make_top_k(/*k=*/10, /*T=*/1.0f, /*seed=*/0);
    const std::array<float, 3> logits = {1.0f, 2.0f, 3.0f};
    bool threw = false;
    try {
        (void)s->sample(std::span<const float>(logits.data(), logits.size()));
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "TopKSampler::sample throws `not yet wired`");
}

}  // namespace

int main() {
    std::fprintf(stderr, "[1bit.cpp smoke test]\n");
    test_chat_template_llama3_parity();
    test_chat_template_short_parity();
    test_chat_template_raw_parity();
    test_sanitize_scrubs_injection();
    test_chat_template_from_str();
    test_router_constructs_with_stub_backend();
    test_router_forward_token_throws_not_yet_wired();
    test_router_load_model_missing_file_throws();
    test_greedy_sampler_argmax();
    test_top_k_sampler_throws_not_yet_wired();
    if (g_failures) {
        std::fprintf(stderr, "\n[1bit.cpp smoke test] %d failure(s)\n",
                     g_failures);
        return 1;
    }
    std::fprintf(stderr, "\n[1bit.cpp smoke test] all passed\n");
    return 0;
}
