#pragma once

// 1bit-agent — abstract tool registry.
//
// The canonical IToolRegistry definition lives in
// `onebit/agent/tools/registry.hpp` (sibling-owned, alongside the
// concrete ToolRegistry). This header is a re-export shim so the
// loop + main + factory sites can `#include "onebit/agent/tools.hpp"`
// without pulling in the entire tool-factory surface.
//
// Coordinated naming: per the comment in tools/registry.hpp, the
// abstract IToolRegistry was supposed to land here under the loop
// owner. We side-step the would-be ODR collision by forwarding to
// the sibling header rather than redeclaring the class.

#include "onebit/agent/tools/registry.hpp"
