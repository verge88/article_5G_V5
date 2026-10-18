#!/usr/bin/env bash
set -euo pipefail
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  ninja-build build-essential flex bison git cmake libsctp-dev libc-ares-dev \
  libgnutls28-dev libgcrypt20-dev libssl-dev libidn11-dev libmongoc-dev libbson-dev \
  libyaml-dev libnghttp2-dev libmicrohttpd-dev libcurl4-gnutls-dev libtins-dev \
  libtalloc-dev meson tcpdump
sudo modprobe sctp
if ! ip link show ogstun >/dev/null 2>&1; then
  sudo ip tuntap add name ogstun mode tun
  sudo ip addr add 10.45.0.1/16 dev ogstun
  sudo ip addr add 2001:db8:cafe::1/48 dev ogstun
  sudo ip link set ogstun up
fi
