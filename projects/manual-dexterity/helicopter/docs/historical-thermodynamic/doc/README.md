# Entropy-Minimizing Control of a Helicopter

One-page technical brief, maintained as a native LaTeX project using the same
IEEEtran conference class, preamble, and bibliography style as `../../delta`.

- `helicopter-entropy-optimal-control.tex`: editable manuscript and equations.
- `references.bib`: the three cited sources and their clickable links.
- `ieee-preamble.tex`: the unchanged Delta project preamble.
- `vendor/IEEEtran/`: the unchanged class, bibliography style, and license files.
- `build.sh`: pdfLaTeX/BibTeX build with PDF integrity and single-page checks.
- `build.ps1`: Windows wrapper for the existing `documents-latex` environment.
- `helicopter-entropy-optimal-control.pdf`: final reading copy.
- `build/`: generated PDF, bibliography, and compiler logs.

The manuscript describes the implemented native entropy NMPC, its assumed
23-state electric helicopter, physical entropy/energy balances, and terminal
completion cost. It distinguishes airborne optimization from PID fallback and
explicit PID landing support. The brief reports five passing paired simulation
cases, small modeled entropy savings, the tracking tradeoff, unsettled handovers,
and missed real-time deadlines. It asserts no hardware efficiency. Its final section prints
the full GitHub URL. Evaluation procedures are in [`../README.md`](../README.md).

It uses US Letter, two columns,
10-point body text, and the standard IEEEtran margins. The PDF is generated
directly from the `.tex` and `.bib` sources; no Python generator is required.

## Build on Windows

From this directory, with the existing LaTeX environment running:

```powershell
.\build.ps1
```

The result is `build/helicopter-entropy-optimal-control.pdf`. After reviewing it,
build and update the reading copy in this documentation directory with:

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
