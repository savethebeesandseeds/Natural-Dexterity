# Transfer and Shared Learning for Atari Control

One-page technical brief, maintained as a native LaTeX project using the same
IEEEtran conference class, preamble, and bibliography style as `../../delta`.

- `atari-transfer-learning.tex`: editable manuscript and equations.
- `references.bib`: cited research with clickable source links.
- `ieee-preamble.tex`: the unchanged Delta project preamble.
- `vendor/IEEEtran/`: unchanged class, bibliography style, and license files.
- `build.sh`: pdfLaTeX/BibTeX build with PDF integrity and single-page checks.
- `build.ps1`: Windows wrapper for the existing `documents-latex` environment.
- `atari-transfer-learning.pdf`: final reading copy.
- `build/`: generated PDF, bibliography, and compiler logs.

The brief describes the implemented approach: pixel-based PPO, feature-only
transfer with fresh target heads and optimizer, a shared encoder with separate
game heads, balanced joint learning, explicit retention checks, and reproducible
native execution. It focuses on method and design rationale; it contains no
experimental results or numerical training settings. It uses US Letter, two
columns, 10-point body text, and standard IEEEtran margins. The PDF is compiled
from `.tex` and `.bib`.

## Build on Windows

From this directory, with the existing LaTeX environment running:

```powershell
.\build.ps1
```

The result is `build/atari-transfer-learning.pdf`. To build and update the
reading copy in the project root:

```powershell
.\build.ps1 -Publish
```

If the existing environment is stopped, its authoritative launcher is:

```powershell
& 'C:\Work\documents\cv.ps1' start
```

The wrapper copies TeX inputs into a unique `/tmp` directory in the existing
managed container, compiles there, and retrieves results. It does not create or
replace containers or change mounts. Failed builds retain their snapshot for
diagnosis and do not replace the reading copy.

## Build in an existing TeX environment

```bash
bash build.sh
```

Requires pdfLaTeX, BibTeX, latexmk, the packages named in `ieee-preamble.tex`,
qpdf, and Poppler's `pdfinfo`. These are installed in the documented
`documents-latex` environment. The vendored class is IEEEtran 1.8b. This brief
is not an IEEE publication.
