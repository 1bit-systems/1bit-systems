#pragma once

// BTCPay Server webhook verification.
//
// BTCPay signs request bodies with HMAC-SHA256 keyed on the shared
// secret configured in the store's webhook settings. The signature is
// sent as hex, prefixed `sha256=`, in the `BTCPay-Sig` header.

#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace onebit::tier_mint::btcpay {

// Constant-time compare of expected vs observed MAC. Returns true iff
// `header` (with or without "sha256=" prefix) matches HMAC-SHA256(secret,
// body) in lowercase hex.
[[nodiscard]] bool verify_signature(std::span<const std::uint8_t> secret,
                                    std::span<const std::uint8_t> body,
                                    std::string_view              header) noexcept;

// Compute the hex MAC used by BTCPay clients / our integration tests.
// Returns "sha256=<hex digest>".
[[nodiscard]] std::string sign_for_test(std::span<const std::uint8_t> secret,
                                        std::span<const std::uint8_t> body);

struct Event {
    std::string event_type;   // "InvoiceSettled", etc.
    std::string invoice_id;
};

// Parse the JSON body produced by BTCPay's webhook. Returns an empty
// invoice_id if absent; callers must check.
[[nodiscard]] Event parse_event(std::string_view body);

} // namespace onebit::tier_mint::btcpay
