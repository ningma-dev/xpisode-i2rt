#!/usr/bin/env bash
# Let a user's processes run the controller at real-time priority, once per machine. Survives
# rebuilding the controller, which a capability on its binary does not. Takes effect after a reboot.
# Check with: python -m xpisode_i2rt.doctor
#
#     sudo scripts/setup_host.sh [user]
set -euo pipefail

user="${1:-${SUDO_USER:-}}"
if [[ $EUID -ne 0 || -z "$user" || "$user" == root ]]; then
  echo "usage: sudo $0 [user]" >&2
  exit 2
fi
id "$user" >/dev/null

# Logins through PAM: SSH and consoles.
file=/etc/security/limits.d/90-xpisode-i2rt.conf
cat >"$file" <<LIMITS
# xpisode-i2rt: SCHED_FIFO and locked memory for the arm controller.
$user - rtprio 90
$user - memlock unlimited
LIMITS
echo "wrote $file"

# A desktop's terminals descend from the user's systemd manager, which PAM's limits do not reach: it
# takes them from its own unit, and hands its children what user.conf says.
dropin=/etc/systemd/system/user@.service.d
mkdir -p "$dropin"
cat >"$dropin/90-xpisode-i2rt.conf" <<UNIT
# xpisode-i2rt: what the user's manager may hand to its children.
[Service]
LimitRTPRIO=90
LimitMEMLOCK=infinity
UNIT
echo "wrote $dropin/90-xpisode-i2rt.conf"

dropin=/etc/systemd/user.conf.d
mkdir -p "$dropin"
cat >"$dropin/90-xpisode-i2rt.conf" <<MANAGER
# xpisode-i2rt: what the user's desktop and terminals start with.
[Manager]
DefaultLimitRTPRIO=90
DefaultLimitMEMLOCK=infinity
MANAGER
echo "wrote $dropin/90-xpisode-i2rt.conf"

systemctl daemon-reload
echo "reboot, then run the doctor in the terminal you run the station from"
