#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="axum-pi-camera"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PI_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd -- "${PI_DIR}/.." && pwd)"
INSTALL_USER="${SUDO_USER:-${USER}}"
INSTALL_GROUP="$(id -gn "${INSTALL_USER}")"
USER_HOME="$(getent passwd "${INSTALL_USER}" | cut -d: -f6)"
VENV_DIR="${AXUM_PI_VENV:-${USER_HOME}/axum-pi-venv}"
SERVICE_TEMPLATE="${PI_DIR}/systemd/${SERVICE_NAME}.service.in"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
ENV_FILE="/etc/default/${SERVICE_NAME}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This installer must run on Raspberry Pi OS/Linux." >&2
  exit 1
fi
if [[ ! -r /proc/device-tree/model ]] || ! tr -d '\0' </proc/device-tree/model | grep -qi "Raspberry Pi"; then
  echo "Raspberry Pi hardware was not detected." >&2
  exit 1
fi
if [[ ! -f "${SERVICE_TEMPLATE}" ]]; then
  echo "Missing service template: ${SERVICE_TEMPLATE}" >&2
  exit 1
fi

sudo apt-get update
sudo apt-get install -y \
  python3-picamera2 \
  python3-opencv \
  python3-venv \
  python3-pip \
  avahi-daemon

if apt-cache show rpicam-apps >/dev/null 2>&1; then
  sudo apt-get install -y rpicam-apps
elif apt-cache show libcamera-apps >/dev/null 2>&1; then
  sudo apt-get install -y libcamera-apps
else
  echo "Neither rpicam-apps nor libcamera-apps is available from configured apt repositories." >&2
  exit 1
fi

if command -v rpicam-hello >/dev/null 2>&1; then
  rpicam-hello --list-cameras
elif command -v libcamera-hello >/dev/null 2>&1; then
  libcamera-hello --list-cameras
else
  echo "Camera utilities are unavailable after package installation." >&2
  exit 1
fi

if [[ ! -x "${VENV_DIR}/bin/python3" ]]; then
  sudo -u "${INSTALL_USER}" python3 -m venv --system-site-packages "${VENV_DIR}"
fi
sudo -u "${INSTALL_USER}" "${VENV_DIR}/bin/python3" -m pip install --upgrade pip
sudo -u "${INSTALL_USER}" "${VENV_DIR}/bin/python3" -m pip install -r "${PI_DIR}/requirements.txt"

if [[ ! -f "${ENV_FILE}" ]]; then
  sudo install -m 0644 "${PI_DIR}/env.example" "${ENV_FILE}"
fi

sed \
  -e "s|@AXUM_USER@|${INSTALL_USER}|g" \
  -e "s|@AXUM_GROUP@|${INSTALL_GROUP}|g" \
  -e "s|@AXUM_REPO@|${REPO_DIR}|g" \
  -e "s|@AXUM_VENV@|${VENV_DIR}|g" \
  "${SERVICE_TEMPLATE}" | sudo tee "${SERVICE_FILE}" >/dev/null
sudo chmod 0644 "${SERVICE_FILE}"

sudo systemctl daemon-reload
sudo systemctl enable --now avahi-daemon.service
sudo systemctl enable --now "${SERVICE_NAME}.service"

"${VENV_DIR}/bin/python3" "${PI_DIR}/verify_pi.py" --url "http://127.0.0.1:5001"

HOSTNAME_VALUE="$(hostname)"
IP_VALUE="$(hostname -I | awk '{print $1}')"
echo
echo "AXUM Pi camera service installed and verified."
echo "Service: sudo systemctl status ${SERVICE_NAME}"
echo "Logs:    journalctl -u ${SERVICE_NAME} -f"
echo "Host:    ${HOSTNAME_VALUE}.local"
echo "IP:      ${IP_VALUE:-not-assigned}"
echo "Laptop config: PI_CAM_URL = \"http://${HOSTNAME_VALUE}.local:5001/stream\""
