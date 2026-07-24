#!/bin/bash
# Install/uninstall the 1bit-agent systemd user service

set -euo pipefail

SERVICE_NAME="1bit-agent"
SERVICE_SRC="$(dirname "$0")/1bit-agent.service"
SERVICE_DEST="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/1bit-agent.service"
# Resolve the actual repo root so the unit works regardless of checkout location (#148).
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

mkdir -p "$(dirname "$SERVICE_DEST")"

case "${1:-install}" in
  install)
    echo "  → Installing $SERVICE_NAME systemd user service (repo: $REPO_ROOT)..."
    # Substitute the templated path with this checkout's actual location (#148).
    sed "s#__REPO_ROOT__#${REPO_ROOT}#g" "$SERVICE_SRC" > "$SERVICE_DEST"
    systemctl --user daemon-reload
    systemctl --user enable "$SERVICE_NAME"
    systemctl --user start "$SERVICE_NAME"
    echo "  ✅ $SERVICE_NAME installed and started"
    echo "  📍 Check: systemctl --user status $SERVICE_NAME"
    ;;

  uninstall)
    echo "  → Removing $SERVICE_NAME systemd user service..."
    systemctl --user stop "$SERVICE_NAME" 2>/dev/null || true
    systemctl --user disable "$SERVICE_NAME" 2>/dev/null || true
    rm -f "$SERVICE_DEST"
    systemctl --user daemon-reload
    echo "  ✅ $SERVICE_NAME removed"
    ;;

  status)
    systemctl --user status "$SERVICE_NAME" 2>&1
    ;;

  *)
    echo "Usage: $0 [install|uninstall|status]"
    exit 1
    ;;
esac
