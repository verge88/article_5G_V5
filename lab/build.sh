#!/usr/bin/env bash
set -euo pipefail
mkdir -p .lab
git clone --branch v2.7.5 --depth 1 https://github.com/open5gs/open5gs.git .lab/open5gs
python lab/patch_open5gs.py .lab/open5gs
meson setup .lab/open5gs/build .lab/open5gs --buildtype=release
targets=(tests/registration/registration)
for nf in nrf scp amf smf upf ausf udm udr pcf nssf bsf sepp; do
  targets+=("src/$nf/open5gs-${nf}d")
done
ninja -C .lab/open5gs/build -j2 "${targets[@]}"
git -C .lab/open5gs diff > .lab/open5gs-instrumentation.patch
tar -czf .lab/open5gs-build.tar.gz .lab/open5gs/build .lab/open5gs-instrumentation.patch
