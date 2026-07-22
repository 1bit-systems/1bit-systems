"""UDP auto-discovery beacon so 1bit Mobile finds this jarvis server on the
LAN without the user typing in an IP address -- matches the
`BeaconListenerService` on port 13305 the app already listens on
(`{"service": "1bit", "hostname": ..., "url": ...}` broadcast datagrams).
"""
import json
import socket
import threading
import time

BEACON_PORT = 13305


def _lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def start_beacon(port, interval=3.0):
    """Start a daemon thread broadcasting this server's address every `interval`s."""
    url = f"http://{_lan_ip()}:{port}"
    hostname = socket.gethostname()
    payload = json.dumps({"service": "1bit", "hostname": hostname, "url": url}).encode()

    def _run():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        while True:
            try:
                s.sendto(payload, ("<broadcast>", BEACON_PORT))
            except Exception:
                pass
            time.sleep(interval)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    print(f"  Beacon: advertising {url} on UDP :{BEACON_PORT}")
    return t
