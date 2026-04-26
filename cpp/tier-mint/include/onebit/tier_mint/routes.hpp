#pragma once

#include "onebit/tier_mint/state.hpp"

#include <httplib.h>

namespace onebit::tier_mint {

// Wire all routes onto `server`. `state` must outlive the server.
//   GET  /v1/health
//   POST /btcpay/webhook
//   GET  /tier/poll/:invoice_id
//   POST /tier/revoke
void build_router(httplib::Server& server, AppState& state);

} // namespace onebit::tier_mint
