# Underactuated Humanoid Locomotion

One-page technical brief, maintained as a native LaTeX project using the same
IEEEtran conference class, preamble, and bibliography style as `../../delta`.

- `humanoid-underactuated-deep-learning.tex`: editable manuscript and equations.
- `references.bib`: the three cited sources and their clickable links.
- `ieee-preamble.tex`: the unchanged Delta project preamble.
- `vendor/IEEEtran/`: the unchanged class, bibliography style, and license files.
- `build.sh`: pdfLaTeX/BibTeX build with PDF integrity and single-page checks.
- `build.ps1`: Windows wrapper for the existing `documents-latex` environment.
- `humanoid-underactuated-deep-learning.pdf`: final reading copy.
- `build/`: generated PDF, bibliography, and compiler logs.

The manuscript preserves the title
"Underactuated Humanoid Locomotion", the abstract's "This brief", and the
subtitle "Learning-based control method for bipedal walking". It uses US Letter,
two columns, 10-point body text, and the standard IEEEtran margins. The PDF is
generated directly from `.tex` and `.bib`; no Python generator is required.

The approach revision describes the native MuJoCo/LibTorch pipeline, a published
walking baseline, distinct imitation and PPO learning routes, causal observation
history, physical and visual gait checks, and bounded experiments. Deliberately
passive-joint motion follows the initial walking task. It includes no measured
values or experimental results; hardware transfer remains future work.

## Build on Windows

From this directory, with the existing LaTeX environment running:

```powershell
.\build.ps1
```

The result is `build/humanoid-underactuated-deep-learning.pdf`. After reviewing
it, build and update the reading copy in the project root with:

```powershell
.\build.ps1 -Publish
```

If the existing environment is stopped, its authoritative launcher is:

```powershell
& 'C:\Work\documents\cv.ps1' start
```

The wrapper copies only TeX inputs into a unique `/tmp` directory in the existing
managed container, compiles there, and retrieves the results. It does not create
or replace containers or change their mounts. Failed builds retain their
snapshot for diagnosis and do not replace the reading copy.

## Build in an existing TeX environment

```bash
bash build.sh
```

Requires pdfLaTeX, BibTeX, latexmk, the packages named in `ieee-preamble.tex`,
qpdf, and Poppler's `pdfinfo`. These are already installed in the documented
`documents-latex` environment. The vendored class is IEEEtran 1.8b. This brief
is not an IEEE publication.
