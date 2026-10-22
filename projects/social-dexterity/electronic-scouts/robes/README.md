# Electronic Scouts / Garment Lab 01

A measurement-driven first sewing draft of the charcoal-and-blue clothing in the two concept images in the parent directory. The design consists of a wrap robe with short drop-shoulder sleeves and optional hood, a blue wrap undertunic, optional tabards, and a short fastening sash. All sizes are parametric. All dimensions are **millimeters**.

The supplied measurements are an **illustrative configuration**, not a verified child size. The cuts have been checked geometrically but have not been sewn or fitted. Make a toile in similar-weight inexpensive fabric before cutting final cloth. The partly visible reference images do not establish the finished hem length; the example length is a design choice.

## Use the pattern

1. Copy `measurements.example.json` to a file named for the wearer and take their actual measurements over the intended underlayers.
2. Edit body dimensions, ease and design controls independently. Do not uniformly scale by age or height.
3. Run the generator. It rebuilds all dependent pieces and allowances together.
4. Read `output/pdf/scout-robe-guide.pdf`, check the print calibration, then cut and baste a toile.
5. Fit through movement, update the measurements/options, regenerate and keep the accepted JSON with the finished garment.

From this folder in PowerShell:

```powershell
.\Generate-Robe.ps1 -Measurements .\measurements.example.json
# Keep a wearer's output separate from the illustrative sample:
.\Generate-Robe.ps1 -Measurements .\measurements.alex.json -Output .\output-alex
```

The launcher uses the bundled Codex Python if available, otherwise `python` on PATH. On another machine, install Python 3 and the one export dependency:

```text
python -m pip install -r requirements.txt
python generate_pattern.py --measurements measurements.example.json --output output
```

Use `--no-tiles` (PowerShell: `-NoTiles`) to omit A4 generation in a fresh output folder. This option refuses a folder containing an existing A4 set, preventing an old size from being mistaken for the current one. The output folder is regenerated in place; use a new folder to retain a previous version. Existing files for previously enabled components may remain in the SVG folder: `pattern.json` and the newly generated PDFs are the authoritative current cut list.

## Files you receive

| File | Purpose |
| --- | --- |
| `output/pdf/scout-robe-guide.pdf` | Design, measurements, cut catalogue, assembly, printing and seam checks. Catalogue diagrams are not to scale. |
| `output/pdf/scout-robe-full-size.pdf` | A4 calibration page followed by one unique piece per custom-sized page at 1:1. Use a plotter or Acrobat Poster at 100%. |
| `output/pdf/scout-robe-a4.pdf` | Already tiled A4 patterns at 1:1, with piece index, row/column labels and 10 mm overlap. |
| `output/svg/*.svg` | Separate 1:1 vector cutting templates with physical dimensions in millimeters. Preserve their document units. |
| `output/cut-list.csv` | Quantities, materials and bounding rectangles for planning. This is not an optimized fabric layout. |
| `output/pattern.json` | Resolved inputs, exact polygon coordinates, edge finishes, derived dimensions and seam checks. |
| `output/parameter-reference.md` | Every input with its meaning, current value, drafting formulas and configuration-specific fitting notices. |

The **solid dark line is the cutting line; allowances are already included**. The dashed blue line is the stitching line on joined edges and the finished boundary on hemmed or bound edges. Read the edge notes: a bound edge intentionally has zero added allowance. Grain arrows align with the selvage; the bias strip has a diagonal grain indication. Mirrored pairs need opposite-handed fabric copies. Full backs are cut as full pieces, not on a fold.

Print the calibration page first at **Actual size / 100%**, with Fit and Shrink disabled. Confirm both sides of the 50 mm square and the 100 mm bar. A4 tile frames are 190 x 250 mm with 10 mm overlap: match the duplicate marks/lines within the overlap. Do not simply butt page edges together. Print only the page ranges for components you plan to sew. Check another 100 mm interval after taping.

## Measurements and design controls

`pattern_geometry.py` provides `DEFAULT_PARAMS` and `PARAM_SPECS`; the JSON example lists every available input. Missing inputs use documented defaults. Unknown keys and invalid values fail instead of silently changing the draft.

| Input family | How to use it |
| --- | --- |
| Chest, hip, waist, upper arm, wrist, hand | Actual circumferences. Hand clearance matters when a cuff must pass over it. |
| Shoulder width | Point-to-point across the back, not neck-to-shoulder. |
| Shoulder to waist | High shoulder beside the neck down to natural waist. |
| Neck circumference | Base of neck; the pattern adds separate neck ease. |
| Head circumference and height | Brow/back-of-head circumference and vertical shoulder-level-to-crown height. Measure these separately from torso size. |
| Robe and tunic lengths | High shoulder beside the neck to desired hem. Check foot clearance on the toile. |
| Sleeve lengths | Shoulder point to desired cuff. The draft subtracts the drop-shoulder extension when calculating the cut sleeve length. |
| Ease | Additional room beyond measured body size. Torso, neck, hood and sleeves have independent controls. |
| Wrap overlap | Extra coverage across the front. Check closure position and sitting coverage while wearing the toile. |
| Hem and seam allowances | Separate values; joining allowances, hem turns and zero-allowance binding edges are handled individually. |
| Component switches | Include or omit hood, undertunic, tabards and sash. The robe is the core component. |

The default silhouette uses short outer sleeves and a slightly longer blue sleeve underneath. The same sleeve family can be lengthened, but a long sleeve needs a fresh reach and cuff fit check. The hood is a simple angular two-panel prototype; its attachment length is matched to the generated robe neckline. Its balance and peripheral view still need a physical fitting.

## Construction intent

- Use soft, matte woven fabric: a soft cotton twill or a linen blend is a reasonable first sample. The reference's appearance is a design guide, not a fabric specification. Prewash and press before laying out.
- Bodies and ordinary accessories follow straight grain. Cut bias strips diagonally to the fabric grain; join short ends with straight seams at the joining allowance, press open, then fold lengthwise into four equal bands. Bind the listed open edges, enclosing their raw edges. The binding strip count includes join reserve for these straight joins.
- Staystitch slanting neck edges. Sew shoulders; attach sleeves flat, matching sleeve midpoint to shoulder seam; close underarms and body sides; then make and attach the hood.
- Join the hood halves along crown/back only. Match its base center to center back and its endpoints to the front neck marks. Do not sew the face opening shut. Clip angular neck seam allowances carefully without cutting the stitching.
- Hem bottoms and cuffs using their marked allowances. Their shaped extensions reflect the flare or taper: do not straighten or truncate them. Follow the seam extensions, turn the marked first 10 mm, then the remaining hem allowance (15 mm in the example). Apply the planned bias finish to front edges and hood face or exposed neckline. Confirm the exact edge treatment from the piece notes and `edge_finishes` in `pattern.json`.
- Sew two-layer accessories right sides together, leaving a turning gap, then turn and close. Fit the sash and short hook-and-loop closure on the wearer; use removable snaps or contained fastening points for optional tabards. Keep the hood cord-free.
- Test overhead and cross-body reach, squatting, kneeling, sitting, long steps, cuff clearance and head turning with the hood up. Adjust and repeat the toile where the fit changes.

The electronics and their carrier are outside this draft. The reference terminal has no reliable dimensions or weight in the folder. Keep its future removable carrier independent of the cloth pattern until those are known.

## Verification and limits

The exporter refuses a failed seam-length check. Checks compare the actual generated polygon segments, including shoulder pairs, body sides, sleeve attachment lengths, sleeve underarms and hood-to-neck attachment. Mathematical agreement does not prove anatomical fit or drape. The included checks exercise multiple configurations and invalid inputs; see `verification.md` for the completed export review.

Fabric quantities depend on usable roll width, grain, nap, shrinkage and the layout of all copies. Cut envelopes do not establish purchase yardage. Lay out the complete required cut count first. Example PDFs and SVGs must be regenerated when measurements change.

## Design references

- The two local images, `Young scout in futuristic setting.png` and `Futuristic scout in a vibrant world.png`, establish color, hood, crossing front, short sleeves and layered silhouette. No physical dimensions were inferred from them.
- [BERNINA / WeAllSew: Sewing Tutorial - Make an Easy Kimono Top](https://weallsew.com/sewing-tutorial-make-easy-kimono-top/) supports flat sleeve assembly, pressing and print calibration. This draft does not reuse its pattern or seam-allowance arithmetic.
- [Pattern Studio 101: 2 Piece Hood Pattern Making](https://www.patternstudio101.com/blog/2-piece-hood-pattern-making) discusses neck drop, overlap extensions and hood attachment.
- [Adobe Acrobat: Print large documents](https://helpx.adobe.com/acrobat/desktop/print-documents/set-up-and-print-pdfs/large-documents.html) documents Poster scale and overlap.
- [US CPSC: Drawstrings in Children's Upper Outerwear](https://www.cpsc.gov/Business--Manufacturing/Business-Education/FAQ?p=2993&tid%5B3001%5D=3001) supports the cord-free hood and use of alternative fastenings. The sash dimensions are design inputs, not a statement of regulatory compliance.

My suggestion for the scout identity: removable mission patches for **observe, measure, build, explain**, plus easy-to-repair seams. The clothing can collect evidence of what a child has learned and mended.
