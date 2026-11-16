#!/usr/bin/env bash
# Copyright 2026 FastLabs Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

baseline_ref=${1:?usage: linux.sh BASE_REVISION RESULTS_DIRECTORY}
baseline=$(git rev-parse --verify --end-of-options "$baseline_ref^{commit}" 2>/dev/null ||
    git rev-parse --verify --end-of-options "refs/remotes/origin/$baseline_ref^{commit}")
results=$(realpath -m "${2:?usage: linux.sh BASE_REVISION RESULTS_DIRECTORY}")
scratch=$(mktemp -d)
baseline_tree="$scratch/baseline"
cpu=$(awk '/Cpus_allowed_list:/ { split($2, cpus, /[-,]/); print cpus[1] }' /proc/self/status)
mkdir -p "$results"

cleanup() {
    git worktree remove --force "$baseline_tree" >/dev/null 2>&1 || true
    rm -rf "$scratch"
}
trap cleanup EXIT

{
    printf 'Current revision: %s\nBaseline library: %s\nPinned CPU: %s\nRUSTFLAGS: %s\n' \
        "$(git rev-parse HEAD)" "$baseline" "$cpu" "${RUSTFLAGS:-<unset>}"
    uname -a
    cat /etc/os-release
    lscpu
    rustc -Vv
    sha256sum Cargo.lock
} | tee "$results/environment.log"

# Keep the harness, workspace settings and dependency lock identical between revisions.
git worktree add --detach "$baseline_tree" HEAD
git -C "$baseline_tree" restore --source="$baseline" --worktree -- bases

build() {
    local revision=$1 tree=$2 binary
    (
        cd "$tree"
        export CARGO_TARGET_DIR="$scratch/target-$revision"
        binary=$(cargo x bench --no-run --json | jq -Rr '
            fromjson? | select(.reason == "compiler-artifact" and .target.name == "throughput") |
            .executable // empty
        ')
        git diff --exit-code -- Cargo.lock
        cp "$binary" "$scratch/$revision"
    )
    test -x "$scratch/$revision"
    taskset -c "$cpu" "$scratch/$revision" --test
}

build before "$baseline_tree"
build after "$PWD"

# Compile both binaries before timing, then alternate their order on the same CPU.
taskset -c "$cpu" "$scratch/after" --list > "$results/scenarios.log"
while read -r scenario _; do
    for sample in 1 2 3 4 5; do
        revisions=(before after)
        if (( sample % 2 == 0 )); then
            revisions=(after before)
        fi
        for revision in "${revisions[@]}"; do
            printf 'revision=%s sample=%s scenario=%s\n' "$revision" "$sample" "$scenario"
            taskset -c "$cpu" "$scratch/$revision" "$scenario" --baseline-only --repeat 1 --raw
        done
    done
done < "$results/scenarios.log" | tee "$results/revisions.log"

taskset -c "$cpu" "$scratch/after" --repeat 5 --raw | tee "$results/peers.log"
