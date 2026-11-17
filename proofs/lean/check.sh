#!/usr/bin/env bash
# Build the proofs, audit them, and run the negative control.
#   1. `lake build` must succeed;
#   2. no `sorry` / `admit` / `axiom` / `native_decide` in the library sources;
#   3. every theorem depends only on propext, Classical.choice, Quot.sound;
#   4. NegativeControl/FalseVariants.lean (false variants of two identities) must FAIL.
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.elan/bin:$PATH"

echo "== build"
lake build 2>&1 | tail -1

echo "== forbidden tokens"
if grep -nE '\bsorry\b|\badmit\b|^axiom |native_decide' EmulationProofs/*.lean; then
  echo "FAIL: forbidden token"; exit 1
fi
echo "none"

echo "== axioms"
ax=$(mktemp --suffix=.lean)
{
  echo "import EmulationProofs"
  for f in EmulationProofs/*.lean; do
    ns=""; grep -q '^namespace EP' "$f" && ns="EP."
    grep -oE '^(theorem|lemma) [A-Za-z0-9_.'"'"']+' "$f" | awk -v ns="$ns" '{print "#print axioms " ns $2}'
  done
} > "$ax"
out=$(lake env lean "$ax" 2>&1)
rm -f "$ax"
n=$(grep -cE "depends on axioms|does not depend on any axioms" <<<"$out" || true)
bad=$(grep "depends on axioms" <<<"$out" | sed 's/.*\[\(.*\)\]/\1/' | tr ',' '\n' | tr -d ' ' \
      | grep -vxE 'propext|Classical.choice|Quot.sound' || true)
if [ -n "$bad" ]; then echo "FAIL: nonstandard axioms: $bad"; exit 1; fi
if grep -qE ":[0-9]+:[0-9]+: error" <<<"$out"; then echo "FAIL: $out"; exit 1; fi
echo "$n theorems, all within {propext, Classical.choice, Quot.sound}"

echo "== negative control (must fail)"
if neg=$(lake env lean NegativeControl/FalseVariants.lean 2>&1); then
  echo "FAIL: the false variants were accepted"; exit 1
fi
errs=$(grep -cE ":[0-9]+:[0-9]+: error" <<<"$neg" || true)
echo "rejected, $errs errors:"
grep -E ":[0-9]+:[0-9]+: error" <<<"$neg"
[ "$errs" -ge 2 ] || { echo "FAIL: expected an error at each false variant"; exit 1; }
echo "OK"
