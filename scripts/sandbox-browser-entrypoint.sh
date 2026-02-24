#!/usr/bin/env bash
set -euo pipefail

export HOME=/tmp/ash-browser-home
export XDG_CONFIG_HOME="${HOME}/.config"
export XDG_CACHE_HOME="${HOME}/.cache"

CDP_PORT="${ASH_BROWSER_CDP_PORT:-${OPENCLAW_BROWSER_CDP_PORT:-9222}}"
HEADLESS="${ASH_BROWSER_HEADLESS:-${OPENCLAW_BROWSER_HEADLESS:-1}}"

mkdir -p "${HOME}" "${HOME}/.chrome" "${XDG_CONFIG_HOME}" "${XDG_CACHE_HOME}"
rm -rf "${HOME}/.chrome/WidevineCdm" >/dev/null 2>&1 || true
rm -f "${HOME}/.chrome/SingletonLock" "${HOME}/.chrome/SingletonCookie" "${HOME}/.chrome/SingletonSocket" >/dev/null 2>&1 || true

if [[ "${HEADLESS}" == "1" ]]; then
  CHROME_ARGS=(
    "--headless=new"
    "--disable-gpu"
  )
else
  CHROME_ARGS=()
fi

CHROME_ARGS+=(
  "--remote-debugging-address=127.0.0.1"
  "--remote-debugging-port=${CDP_PORT}"
  "--user-data-dir=${HOME}/.chrome"
  "--no-first-run"
  "--no-default-browser-check"
  "--disable-dev-shm-usage"
  "--disable-background-networking"
  "--disable-component-update"
  "--disable-features=Translate,MediaRouter"
  "--disable-breakpad"
  "--disable-crash-reporter"
  "--metrics-recording-only"
  "--disable-session-crashed-bubble"
  "--hide-crash-restore-bubble"
  "--password-store=basic"
  "--no-sandbox"
)

chromium "${CHROME_ARGS[@]}" about:blank &

wait -n
