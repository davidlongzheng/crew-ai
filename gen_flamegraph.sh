#!/bin/bash
set -euxo pipefail

./build/test_rollout & sample $! 1000 -file ../out.sample
../Flamegraph/stackcollapse-sample.awk ../out.sample > ../out.folded
../Flamegraph/flamegraph.pl ../out.folded > ../flamegraph.svg
open ../flamegraph.svg