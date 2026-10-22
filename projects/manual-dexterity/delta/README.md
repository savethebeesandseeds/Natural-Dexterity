# Delta Robot with Damped Articulations

One-page English technical summary of Santiago Restrepo Ruiz's undergraduate
thesis, *Construccion de un prototipo de robot Delta con articulaciones
amortiguadas*, Universidad del Quindio, November 2017.

- `delta-robot-summary.pdf`: final reading copy.
- `delta-robot-summary.tex`: editable manuscript and author block.
- `references.bib`: original-thesis reference.
- `assets/`: the supplied prototype photograph and CAD rendering, displayed
  side by side at their original aspect ratios; the image files are unchanged.
- `ieee-preamble.tex` and `vendor/`: formatting dependency reused from the
  requested Cryptographic Electronics example, without modifying IEEEtran.
- `build.sh`: compilation and single-page verification.

Formatting: IEEEtran 1.8b, conference, US Letter, two columns, normal 10-point
body, default margins. This summary is not an IEEE publication.

## Rebuild with the existing container

The existing `documents-latex` environment is maintained by
`C:\Work\documents\README.md`, `cv.ps1`, and `setup.sh`. It mounts
`C:\Work\documents` at `/workspace`. With that managed container running:

```powershell
docker exec documents-latex bash /workspace/projects/delta_robot_summary/build.sh
```

The checked result is `build/delta-robot-summary.pdf`. Copy it to the project
root after reviewing any edits. No new image, container, mount, volume, or
package is required.

## Editorial basis

The full 126-page repository PDF was extracted for review. Construction and
trajectory figures were visually inspected. Primary source:
https://github.com/savethebeesandseeds/robot_delta_doc/blob/main/delta-robot-damped-articulations-restrepo-2017.pdf

Key printed-page references (PDF page number = printed page + 2):

- pp. 21-24: objectives and stepper-prototype/DC-model distinction.
- pp. 25-33: mechanical topology and nominal geometric assumptions.
- pp. 49-66: trajectory control, local linearization, and state observer.
- pp. 67-95: subsystem models, kinematics, Jacobians, and dynamics.
- pp. 60-61: controller plots explicitly exclude the observer.
- pp. 97-98: construction photos and link to a motion demonstration.
- pp. 101-109: evolutionary geometry search and approximate workspace.
- pp. 111-119: Lagrangian formulations and simulation equations.

The title page dates the thesis November 2017; the distributed PDF metadata
records February 2019 generation. The summary cites the title-page date.
It distinguishes constructed hardware, analytical control work, and proposed
measurement needs. The motion video was not independently evaluated. No
payload, accuracy, timing, or measured vibration-reduction claim is added.

The current revision incorporates two images supplied by the author on
2026-09-14 and replaces dimensions and component inventories with a fuller
account of model complexity. Five sections cover coupled geometry,
compliance states, dynamic formulation and design search, control and
estimation, and validation. The compact constraint equation restates the
nominal closure and its velocity relation using summary notation; it does
not claim an additional result from the thesis.

The explicit differential kinematics use the zero-compression rigid
approximation. Appendix D is described as an extended constrained framework
because some energy expressions are unfinished; neither a fully solved
compliant model nor experimental closed-loop validation is claimed.
The PDF remains one page at the template's standard font size and margins.
