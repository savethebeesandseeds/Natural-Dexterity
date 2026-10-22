#!/usr/bin/env bash
# Compile with installed Debian tools and the unmodified vendored IEEEtran.
set -euo pipefail
paper_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$paper_dir"
build_dir=${1:-"$paper_dir/build"}
mkdir -p "$build_dir"
build_dir=$(CDPATH= cd -- "$build_dir" && pwd)
export TEXINPUTS="$paper_dir/vendor/IEEEtran//:${TEXINPUTS:-}"
export BSTINPUTS="$paper_dir/vendor/IEEEtran/bibtex//:${BSTINPUTS:-}"
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error \
  -outdir="$build_dir" humanoid-underactuated-deep-learning.tex
qpdf --check "$build_dir/humanoid-underactuated-deep-learning.pdf"
pages=$(pdfinfo "$build_dir/humanoid-underactuated-deep-learning.pdf" | awk '/^Pages:/ {print $2}')
test "$pages" = 1 || { printf 'Expected one page; obtained %s\n' "$pages" >&2; exit 1; }
