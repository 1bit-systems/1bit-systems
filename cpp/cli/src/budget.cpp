#include "onebit/cli/budget.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <string>

namespace onebit::cli {

std::optional<std::uint64_t>
parse_meminfo_kib(std::string_view meminfo, std::string_view key) noexcept
{
    std::size_t pos = 0;
    while (pos < meminfo.size()) {
        const std::size_t eol = meminfo.find('\n', pos);
        const std::size_t end = eol == std::string_view::npos ? meminfo.size() : eol;
        std::string_view line = meminfo.substr(pos, end - pos);
        pos = (eol == std::string_view::npos) ? meminfo.size() : eol + 1;

        if (line.size() < key.size() || line.substr(0, key.size()) != key) continue;
        std::string_view rest = line.substr(key.size());
        // Trim ':' and whitespace
        while (!rest.empty() && (rest.front() == ':' || rest.front() == ' '
                                  || rest.front() == '\t')) {
            rest.remove_prefix(1);
        }
        if (rest.empty()) return std::nullopt;
        std::uint64_t v = 0;
        std::size_t i = 0;
        while (i < rest.size() && rest[i] >= '0' && rest[i] <= '9') {
            v = v * 10 + static_cast<std::uint64_t>(rest[i] - '0');
            ++i;
        }
        if (i == 0) return std::nullopt;
        return v;
    }
    return std::nullopt;
}

std::string fmt_bytes(std::uint64_t n)
{
    constexpr std::uint64_t K = 1024ULL;
    constexpr std::uint64_t M = K * 1024ULL;
    constexpr std::uint64_t G = M * 1024ULL;
    if (n >= G) return fmt::format("{:.1f} GB", static_cast<double>(n) / static_cast<double>(G));
    if (n >= M) return fmt::format("{:.1f} MB", static_cast<double>(n) / static_cast<double>(M));
    if (n >= K) return fmt::format("{:.1f} KB", static_cast<double>(n) / static_cast<double>(K));
    return fmt::format("{} B", n);
}

bool looks_like_halo_service(std::string_view comm) noexcept
{
    static constexpr std::array<std::string_view, 8> H = {
        "bitnet_decode", "whisper-server", "sd-server", "kokoro",
        "lemond", "halo-", "1bit-", "halo-server",
    };
    for (auto h : H) if (comm.find(h) != std::string_view::npos) return true;
    return false;
}

std::uint64_t BudgetSnapshot::budget_for_next_model() const noexcept
{
    const std::uint64_t free = gtt_free();
    const std::uint64_t ram_free =
        mem_available > COMPOSITOR_RESERVE_BYTES
            ? mem_available - COMPOSITOR_RESERVE_BYTES
            : 0;
    return std::min(free, ram_free);
}

std::string BudgetSnapshot::render() const
{
    std::string s;
    s += fmt::format("GTT   {:>7} / {:>7}  ({} free)\n",
                     fmt_bytes(gtt_used), fmt_bytes(gtt_total), fmt_bytes(gtt_free()));
    s += fmt::format("VRAM  {:>7} / {:>7}  (stolen BAR — not the model pool)\n",
                     fmt_bytes(vram_used), fmt_bytes(vram_total));
    s += fmt::format("RAM   {:>7} / {:>7}  available\n",
                     fmt_bytes(mem_available), fmt_bytes(mem_total));
    s += fmt::format("next model budget  ≈ {} (min(GTT free, RAM avail − 4 GB reserve))\n",
                     fmt_bytes(budget_for_next_model()));
    s += "---- halo services ----\n";
    for (const auto& svc : services) {
        s += fmt::format("  {:<28} {:>7}\n", svc.name, fmt_bytes(svc.rss_kib * 1024ULL));
    }
    return s;
}

}  // namespace onebit::cli
