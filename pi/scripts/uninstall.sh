#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="axum-pi-camera"

sudo systemctl disable --now "${SERVICE_NAME}.service" 2>/dev/null || true
sudo rm -f "/etc/systemd/system/${SERVICE_NAME}.service"
sudo rm -f "/etc/default/${SERVICE_NAME}"
sudo systemctl daemon-reload
sudo systemctl reset-failed

echo "Removed ${SERVICE_NAME}. The repository, virtual environment, and captured data were not deleted."
