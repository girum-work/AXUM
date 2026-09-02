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

CAMERA_TOOL="$(command -v rpicam-hello || command -v libcamera-hello || true)"
if [[ -z "${CAMERA_TOOL}" ]]; then
  echo "Camera utilities are unavailable after package installation." >&2
  exit 1
fi

# --list-cameras exits 0 whether or not a sensor answered, so the exit code
# says nothing. The presence of an "0 : <sensor>" line is the real signal.
CAMERA_LIST="$("${CAMERA_TOOL}" --list-cameras 2>&1 || true)"
echo "${CAMERA_LIST}"
if ! grep -qE '^[[:space:]]*[0-9]+[[:space:]]*:' <<<"${CAMERA_LIST}"; then
  if [[ "${AXUM_ALLOW_NO_CAMERA:-0}" == "1" ]]; then
    echo
    echo "WARNING: no camera sensor detected. Installing the service anyway" >&2
    echo "because AXUM_ALLOW_NO_CAMERA=1. It will run degraded and /status will" >&2
    echo "return 503 until a sensor is attached." >&2
    SKIP_VERIFY=1
  else
    echo
    echo "No camera sensor detected on the CSI port." >&2
    echo "Power the Pi down before reseating the ribbon: contacts toward the" >&2
    echo "board, blue tab toward the USB sockets. Re-run once" >&2
    echo "'${CAMERA_TOOL##*/} --list-cameras' lists a sensor." >&2
    echo "To install the service before the camera arrives, re-run with" >&2
    echo "AXUM_ALLOW_NO_CAMERA=1." >&2
    exit 1
  fi
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
sudo systemctl enable "${SERVICE_NAME}.service"
# restart, not enable --now: on a re-run after git pull the unit is already
# active, and --now would leave the previous code running.
sudo systemctl restart "${SERVICE_NAME}.service"

if [[ "${SKIP_VERIFY:-0}" == "1" ]]; then
  echo
  echo "Skipping verification: there is no camera to verify against."
else
  "${VENV_DIR}/bin/python3" "${PI_DIR}/verify_pi.py" --url "http://127.0.0.1:5001"
fi

HOSTNAME_VALUE="$(hostname)"
IP_VALUE="$(hostname -I | awk '{print $1}')"
echo
echo "AXUM Pi camera service installed."
echo "Service: sudo systemctl status ${SERVICE_NAME}"
echo "Logs:    journalctl -u ${SERVICE_NAME} -f"
echo "Host:    ${HOSTNAME_VALUE}.local"
echo "IP:      ${IP_VALUE:-not-assigned}"
echo "Laptop config: PI_CAM_URL = \"http://${HOSTNAME_VALUE}.local:5001/stream\""
