#include "onebit/cli/install_tracker.hpp"

#include "onebit/cli/proc.hpp"

#include <fmt/core.h>
#include <iostream>
#include <mutex>
#include <system_error>

namespace onebit::cli {

struct InstallTracker::Impl {
    mutable std::mutex            mu;
    std::vector<InstallAction>    actions;
};

// Special members defined here AFTER `Impl` is complete — ISO C++ Core
// Guidelines I.27. Pulling these into the header would force a partial
// `Impl` definition for the unique_ptr's deleter and trip on incomplete
// type errors at every translation unit that includes the header.
InstallTracker::InstallTracker()
    : impl_(std::make_unique<Impl>()) {}

InstallTracker::~InstallTracker() = default;
InstallTracker::InstallTracker(InstallTracker&&) noexcept = default;
InstallTracker& InstallTracker::operator=(InstallTracker&&) noexcept = default;

void InstallTracker::record(InstallAction action)
{
    std::lock_guard<std::mutex> lk(impl_->mu);
    impl_->actions.push_back(std::move(action));
}

std::vector<InstallAction> InstallTracker::actions() const
{
    std::lock_guard<std::mutex> lk(impl_->mu);
    return impl_->actions;
}

bool InstallTracker::empty() const noexcept
{
    std::lock_guard<std::mutex> lk(impl_->mu);
    return impl_->actions.empty();
}

std::size_t InstallTracker::size() const noexcept
{
    std::lock_guard<std::mutex> lk(impl_->mu);
    return impl_->actions.size();
}

void InstallTracker::best_effort_revert()
{
    std::vector<InstallAction> drained;
    {
        std::lock_guard<std::mutex> lk(impl_->mu);
        drained.swap(impl_->actions);
    }

    if (drained.empty()) {
        std::cout
            << "    (nothing to revert — install failed before any side-effect landed)\n";
        return;
    }
    std::cout << "    atomic revert: " << drained.size() << " action(s) to undo\n";

    // LIFO — mirror actual stack-unwind order.
    for (auto it = drained.rbegin(); it != drained.rend(); ++it) {
        std::visit([](const auto& a) {
            using T = std::decay_t<decltype(a)>;
            if constexpr (std::is_same_v<T, ActionEnabledUnit>) {
                std::cout << "      - disabling user unit " << a.unit << "\n";
                (void)run_inherit({"systemctl", "--user", "disable", "--now", a.unit});
            } else if constexpr (std::is_same_v<T, ActionCopiedFile>) {
                std::cout << "      - removing " << a.path.string() << "\n";
                std::error_code ec;
                std::filesystem::remove(a.path, ec);
            } else if constexpr (std::is_same_v<T, ActionCopiedSudo>) {
                std::cout << "      - removing (sudo) " << a.path.string() << "\n";
                (void)run_inherit({"sudo", "rm", "-f", a.path.string()});
            }
        }, *it);
    }
}

}  // namespace onebit::cli
