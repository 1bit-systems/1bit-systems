// beacon.h — UDP LAN discovery beacon.
// C++ port of jarvis/beacon.py (recovered from git history at
// c252174aa~1). Broadcasts {"service":"1bit","hostname":...,"url":...} on
// UDP port 13305 every ~3s so the 1bit Mobile app's BeaconListenerService
// can auto-discover this server without manual IP entry.
#pragma once

namespace jarvis {

constexpr int kBeaconPort = 13305;

// Starts a detached background thread that broadcasts the beacon
// indefinitely. Safe to call once at server startup; does not block.
void start_beacon(int server_port, double interval_seconds = 3.0);

} // namespace jarvis
