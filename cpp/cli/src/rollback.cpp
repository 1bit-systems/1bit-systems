#include "onebit/cli/rollback.hpp"

#include "onebit/cli/oobe_error.hpp"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>

namespace onebit::cli {

std::vector<SnapperEntry> parse_snapper_list(std::string_view stdout_text)
{
    std::vector<SnapperEntry> out;
    std::size_t pos = 0;
    while (pos < stdout_text.size()) {
        const auto eol = stdout_text.find('\n', pos);
        const auto end = eol == std::string_view::npos ? stdout_text.size() : eol;
        std::string_view line = stdout_text.substr(pos, end - pos);
        pos = (eol == std::string_view::npos) ? stdout_text.size() : eol + 1;

        // Trim leading/trailing ws.
        while (!line.empty() && (line.front() == ' ' || line.front() == '\t')) line.remove_prefix(1);
        while (!line.empty() && (line.back()  == ' ' || line.back()  == '\t')) line.remove_suffix(1);
        if (line.empty()) continue;
        if (line.front() == '#' || line.front() == '-') continue;

        // Split on '|', trim each cell.
        std::vector<std::string> cols;
        std::size_t s = 0;
        while (s <= line.size()) {
            const auto bar = line.find('|', s);
            std::string_view cell = (bar == std::string_view::npos)
                ? line.substr(s)
                : line.substr(s, bar - s);
            while (!cell.empty() && (cell.front() == ' ' || cell.front() == '\t')) cell.remove_prefix(1);
            while (!cell.empty() && (cell.back()  == ' ' || cell.back()  == '\t')) cell.remove_suffix(1);
            cols.emplace_back(cell);
            if (bar == std::string_view::npos) break;
            s = bar + 1;
        }
        if (cols.size() < 2) continue;

        // Parse first cell as integer.
        const std::string& head = cols.front();
        if (head.empty()) continue;
        std::uint32_t n = 0;
        bool ok = true;
        for (char c : head) {
            if (c < '0' || c > '9') { ok = false; break; }
            n = n * 10 + static_cast<std::uint32_t>(c - '0');
        }
        if (!ok) continue;

        // Find the description column carrying the halo label, else last cell.
        std::string desc;
        for (const auto& c : cols) {
            if (c.find(HALO_PREINSTALL_LABEL) != std::string::npos) {
                desc = c;
                break;
            }
        }
        if (desc.empty()) desc = cols.back();
        out.push_back(SnapperEntry{n, std::move(desc)});
    }
    return out;
}

std::optional<std::uint32_t>
pick_latest_preinstall(const std::vector<SnapperEntry>& entries)
{
    std::optional<std::uint32_t> best;
    for (const auto& e : entries) {
        if (e.description.find(HALO_PREINSTALL_LABEL) == std::string::npos) continue;
        if (!best || e.number > *best) best = e.number;
    }
    return best;
}

std::expected<void, Error>
run_with_snapper(Snapper& snapper,
                 std::optional<std::uint32_t> snapshot,
                 bool /*yes*/)
{
    if (!snapper.available()) {
        std::cout << "\nerror: rollback\n" << OobeError::snapper_absent() << '\n';
        return std::unexpected(Error::precondition("snapper not installed"));
    }
    auto entries = snapper.list();
    if (!entries) return std::unexpected(entries.error());

    std::uint32_t number = 0;
    if (snapshot) {
        number = *snapshot;
    } else {
        const auto picked = pick_latest_preinstall(*entries);
        if (!picked) {
            std::cout << "\nerror: rollback\n"
                      << OobeError::no_rollback_candidate() << '\n';
            return std::unexpected(Error::not_found(
                "no .halo-preinstall snapshot found"));
        }
        number = *picked;
    }
    auto it = std::find_if(entries->begin(), entries->end(),
                           [&](const SnapperEntry& e) { return e.number == number; });
    SnapperEntry plan = (it != entries->end())
        ? *it
        : SnapperEntry{number, "(no description — explicit snapshot)"};

    std::cout << "rollback plan:\n"
              << "  snapshot : " << plan.number << '\n'
              << "  label    : " << plan.description << '\n'
              << "  command  : sudo snapper -c root rollback " << plan.number << '\n';

    // We deliberately skip the interactive confirm in the testable core —
    // the CLI driver in main.cpp prompts before calling this when needed.
    return snapper.rollback(number);
}

}  // namespace onebit::cli
