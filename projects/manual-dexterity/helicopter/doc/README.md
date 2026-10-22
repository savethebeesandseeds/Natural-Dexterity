# Minimum-entropy H-infinity control of a helicopter

One-page technical brief maintained as native LaTeX with the same IEEEtran
conference class, preamble, and bibliography style as `../../delta`.

The manuscript now describes the user's intended Mustafa–Glover minimum-entropy
H-infinity objective, central DGKF synthesis, and the required nonlinear flight
evaluation. It explicitly distinguishes frequency-domain control entropy from
thermodynamic entropy. **The native corrected-method report passes all 15 trials, with separate linear synthesis checks.** The complete
subscription book has not been accessed; the contract records the accessible
primary papers used to check the equations.

The earlier thermodynamic-NMPC manuscript and published PDF were copied to
[the historical archive](../docs/historical-thermodynamic/README.md). Their passed
simulations are not evidence for the corrected method. During this revision the
published PDF may still contain the old text; rebuild and visually review it
before treating it as the current reading copy.

- `helicopter-entropy-optimal-control.tex`: current editable manuscript.
- `references.bib`: book and primary control papers with clickable links.
- `ieee-preamble.tex`: unchanged shared preamble.
- `vendor/IEEEtran/`: unchanged class, bibliography style, and license.
- `build.sh`: pdfLaTeX/BibTeX build with integrity and single-page checks.
- `build.ps1`: wrapper for the existing `documents-latex` environment.
- `helicopter-entropy-optimal-control.pdf`: published reading copy, updated after review.
- `build/`: generated PDF, bibliography, and compiler logs.

The brief retains US Letter, two columns, 10-point body text, and standard
IEEEtran margins. It prints the literal GitHub repository URL in the last section.
The PDF is generated from TeX/BibTeX; no Python generator is required.
This is not an IEEE publication.

## Build on Windows

From this directory, with the existing LaTeX environment running:

```powershell
.\build.ps1
```

Review `build/helicopter-entropy-optimal-control.pdf`, then publish with:

```powershell
.\build.ps1 -Publish
```

If the existing environment is stopped, its authoritative launcher is:

```powershell
& 'C:\Work\documents\cv.ps1' start
```

The wrapper copies TeX inputs into a unique `/tmp` directory in the existing
managed container, compiles, and retrieves the results. It neither creates nor
replaces containers or changes mounts. Failed builds retain their diagnostic
snapshot and do not replace the published copy.

## Build in an existing TeX environment

```bash
bash build.sh
```

Requires pdfLaTeX, BibTeX, latexmk, the preamble's packages, qpdf, and Poppler's
pdfinfo, already installed in the documented `documents-latex` environment.
The vendored class is IEEEtran 1.8b.
