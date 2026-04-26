#pragma once

// Anchor #10 — atomic-on-failure install tracker.
//
// Thin pImpl class that records every state change a `1bit install` step
// makes (systemd units enabled, files copied) and, if a subsequent step
// fails, best-effort reverts them in LIFO order. NOT a true transaction —
// that's what `1bit rollback` (snapper) is for; this is a "leave the box
// reasonable" cleanup pass.

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace onebit::cli {

struct ActionEnabledUnit  { std::string unit; };
struct ActionCopiedFile   { std::filesystem::path path; };
struct ActionCopiedSudo   { std::filesystem::path path; };

using InstallAction = std::variant<ActionEnabledUnit,
                                   ActionCopiedFile,
                                   ActionCopiedSudo>;

class InstallTracker {
public:
    // Five special members declared in the header so users get a
    // complete-type Tracker without seeing Impl. Defined `= default` in
    // the .cpp AFTER `Impl` is complete (ISO C++ Core Guidelines I.27).
    InstallTracker();
    ~InstallTracker();
    InstallTracker(const InstallTracker&)            = delete;
    InstallTracker& operator=(const InstallTracker&) = delete;
    InstallTracker(InstallTracker&&) noexcept;
    InstallTracker& operator=(InstallTracker&&) noexcept;

    void record(InstallAction action);

    // Snapshot — returns by value so callers don't hold internal locks.
    [[nodiscard]] std::vector<InstallAction> actions() const;

    // True iff zero recorded actions remain.
    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;

    // LIFO best-effort revert. Drains the action log on success — a
    // second call is a no-op.
    void best_effort_revert();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace onebit::cli
