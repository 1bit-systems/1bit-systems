#pragma once

// Static-asset blobs for the 1bit-landing marketing page.
//
// Embedding strategy: at configure time `cmake/embed_assets.cmake` reads
// the files under `cpp/landing/assets/` and emits a generated `.cpp` that
// stores each one as a hex-decoded byte array. We chose this over C++23
// `#embed` (compiler version drift across Strix Halo + Ryzen builds) and
// over `configure_file` raw-string substitution (HTML+JS contains every
// punctuation string a delimiter could collide with). Hex bytes are
// boring and always work.

#include <string_view>

namespace onebit::landing::assets {

extern const std::string_view INDEX_HTML;
extern const std::string_view STYLE_CSS;
extern const std::string_view LOGO_SVG;

} // namespace onebit::landing::assets
