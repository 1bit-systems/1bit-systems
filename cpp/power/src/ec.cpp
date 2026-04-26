#include "onebit/power/ec.hpp"

#include <algorithm>
#include <charconv>
#include <fstream>
#include <sstream>
#include <system_error>

namespace onebit::power {

namespace {

[[nodiscard]] std::string trim(std::string s)
{
    auto not_ws = [](unsigned char c) {
        return c != ' ' && c != '\t' && c != '\r' && c != '\n';
    };
    auto first = std::find_if(s.begin(), s.end(), not_ws);
    auto last  = std::find_if(s.rbegin(), s.rend(), not_ws).base();
    if (first >= last) return {};
    return std::string{first, last};
}

[[nodiscard]] Result<std::string> read_trim(const std::filesystem::path& root,
                                            std::initializer_list<std::string_view> rel)
{
    std::filesystem::path p = root;
    for (auto part : rel) p /= part;
    std::ifstream f{p};
    if (!f) {
        std::ostringstream os; os << "read " << p.string();
        return Status::fail(Error::IoError, os.str());
    }
    std::ostringstream buf; buf << f.rdbuf();
    return trim(buf.str());
}

[[nodiscard]] Status write_str(const std::filesystem::path& root,
                               std::initializer_list<std::string_view> rel,
                               std::string_view val)
{
    std::filesystem::path p = root;
    for (auto part : rel) p /= part;
    std::ofstream f{p, std::ios::binary | std::ios::trunc};
    if (!f) {
        std::ostringstream os; os << "write " << val << " to " << p.string();
        // Real ENOENT vs EACCES split would need <errno> from a syscall path.
        // Sysfs writes from non-root return EACCES; missing nodes return ENOENT.
        return Status::fail(Error::IoError, os.str());
    }
    f << val;
    if (!f.good()) {
        std::ostringstream os; os << "short write to " << p.string();
        return Status::fail(Error::IoError, os.str());
    }
    return Status::success();
}

template <class Int>
[[nodiscard]] std::optional<Int> parse_int(std::string_view s)
{
    Int v{};
    auto* first = s.data();
    auto* last  = s.data() + s.size();
    auto [ptr, ec] = std::from_chars(first, last, v);
    if (ec != std::errc{} || ptr != last) return std::nullopt;
    return v;
}

} // namespace

Status validate_fan_id(std::uint8_t id)
{
    if (id < 1 || id > 3) {
        std::ostringstream os; os << "fan id must be 1..=3, got " << static_cast<int>(id);
        return Status::fail(Error::InvalidArgument, os.str());
    }
    return Status::success();
}

std::vector<std::uint8_t> parse_curve_csv(std::string_view s)
{
    std::vector<std::uint8_t> out;
    std::size_t i = 0;
    while (i < s.size()) {
        std::size_t j = s.find(',', i);
        if (j == std::string_view::npos) j = s.size();
        std::string_view tok = s.substr(i, j - i);
        // trim
        while (!tok.empty() && (tok.front() == ' ' || tok.front() == '\t')) tok.remove_prefix(1);
        while (!tok.empty() && (tok.back()  == ' ' || tok.back()  == '\t' || tok.back() == '\n' || tok.back() == '\r'))
            tok.remove_suffix(1);
        if (auto v = parse_int<unsigned>(tok); v && *v <= 255) {
            out.push_back(static_cast<std::uint8_t>(*v));
        }
        i = j + 1;
    }
    return out;
}

bool EcBackend::available() const noexcept
{
    std::error_code ec;
    auto p = root_ / "apu" / "power_mode";
    return std::filesystem::is_regular_file(p, ec);
}

Result<std::int32_t> EcBackend::temp_c() const
{
    auto r = read_trim(root_, {"temp1", "temp"});
    if (!r) return r.status();
    if (auto v = parse_int<std::int32_t>(r.value()); v) return *v;
    return Status::fail(Error::ParseError, "parse temp `" + r.value() + "`");
}

Result<std::string> EcBackend::power_mode() const
{
    return read_trim(root_, {"apu", "power_mode"});
}

Status EcBackend::set_power_mode(std::string_view mode) const
{
    static constexpr std::array<std::string_view, 3> VALID{"quiet", "balanced", "performance"};
    if (std::find(VALID.begin(), VALID.end(), mode) == VALID.end()) {
        std::ostringstream os;
        os << "power_mode must be one of [quiet,balanced,performance], got `" << mode << "`";
        return Status::fail(Error::InvalidArgument, os.str());
    }
    return write_str(root_, {"apu", "power_mode"}, mode);
}

Result<FanSnapshot> EcBackend::fan(std::uint8_t id) const
{
    if (auto s = validate_fan_id(id); !s) return s;
    std::string name = "fan" + std::to_string(static_cast<int>(id));

    FanSnapshot fs;
    fs.id = id;

    if (auto r = read_trim(root_, {name, "rpm"}); r) {
        fs.rpm = parse_int<std::uint32_t>(r.value()).value_or(0);
    } else {
        return r.status();
    }
    if (auto r = read_trim(root_, {name, "mode"}); r) {
        fs.mode = r.value();
    } else {
        return r.status();
    }
    if (auto r = read_trim(root_, {name, "level"}); r) {
        unsigned lv = parse_int<unsigned>(r.value()).value_or(0);
        fs.level = static_cast<std::uint8_t>(std::min<unsigned>(lv, 255));
    } else {
        return r.status();
    }
    if (auto r = read_trim(root_, {name, "rampup_curve"}); r) {
        fs.rampup = parse_curve_csv(r.value());
    } else {
        return r.status();
    }
    if (auto r = read_trim(root_, {name, "rampdown_curve"}); r) {
        fs.rampdown = parse_curve_csv(r.value());
    } else {
        return r.status();
    }
    return fs;
}

Status EcBackend::set_fan_mode(std::uint8_t id, std::string_view mode) const
{
    if (auto s = validate_fan_id(id); !s) return s;
    static constexpr std::array<std::string_view, 3> VALID{"auto", "fixed", "curve"};
    if (std::find(VALID.begin(), VALID.end(), mode) == VALID.end()) {
        std::ostringstream os;
        os << "fan mode must be one of [auto,fixed,curve], got `" << mode << "`";
        return Status::fail(Error::InvalidArgument, os.str());
    }
    std::string fan = "fan" + std::to_string(static_cast<int>(id));
    return write_str(root_, {fan, "mode"}, mode);
}

Status EcBackend::set_fan_level(std::uint8_t id, std::uint8_t level) const
{
    if (auto s = validate_fan_id(id); !s) return s;
    if (level > 5) {
        std::ostringstream os;
        os << "fan level must be 0..=5, got " << static_cast<int>(level);
        return Status::fail(Error::InvalidArgument, os.str());
    }
    std::string fan = "fan" + std::to_string(static_cast<int>(id));
    std::string val = std::to_string(static_cast<int>(level));
    return write_str(root_, {fan, "level"}, val);
}

Status EcBackend::set_fan_curve(std::uint8_t id,
                                CurveDir direction,
                                const std::array<std::uint8_t, 5>& curve) const
{
    if (auto s = validate_fan_id(id); !s) return s;
    std::string_view file = direction == CurveDir::Rampup ? "rampup_curve" : "rampdown_curve";

    std::ostringstream os;
    for (std::size_t i = 0; i < curve.size(); ++i) {
        if (i) os << ',';
        os << static_cast<int>(curve[i]);
    }
    std::string fan = "fan" + std::to_string(static_cast<int>(id));
    return write_str(root_, {fan, file}, os.str());
}

EcSnapshot EcBackend::snapshot() const
{
    EcSnapshot s;
    if (auto r = temp_c();      r) s.temp_c     = r.value();
    if (auto r = power_mode();  r) s.power_mode = r.value();
    for (std::uint8_t i = 1; i <= 3; ++i) {
        if (auto r = fan(i); r) s.fans.push_back(r.value());
    }
    return s;
}

} // namespace onebit::power
