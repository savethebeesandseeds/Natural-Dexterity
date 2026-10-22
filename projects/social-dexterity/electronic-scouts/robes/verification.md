# Verification / 5 September 2026

The example is a checked geometric prototype, not a sewn or fitted garment.

## Geometry

`check_geometry.py` passed **131 valid configurations**, covering **1,385 generated piece geometries**. It exercises smaller/larger measurement sets, every combination of optional components, allowance and design limits, and 100 deterministic varied drafts. Seven invalid input cases were rejected.

Checks inspect actual output geometry: simple positive cut/seam polygons; joining seam lengths; sufficient binding after joins; and hem seam endpoints folded across both generated fold lines onto the original body/sleeve seam. The latter catches tapered cuffs with an undersized turnback. Sleeve grain runs along the arm.

The default example contains **11 unique templates and 13 passing seam matches**: shoulders, sides, sleeve attachments, underarms, sleeve midpoint divisions, complete hood base and hood shoulder-notch divisions.

Run again after changing the drafting code:

```text
python check_geometry.py
```

## Exports

- Sewing guide: **13 A4 pages**. All piece notes and mark keys are included.
- Master pattern: **12 pages**, comprising one A4 calibration page and 11 custom-sized pattern pages.
- A4 pattern: **71 pages**, comprising calibration/index plus 70 tiles. Page count changes with the measurements and selected components.
- SVG: **11 templates**, with physical millimeter dimensions matching their viewBox units.

An independent export review compared each master outline against `pattern.json`: maximum dimensional difference was below **0.00018 mm**, from PDF coordinate serialization. All outlines fit within their pages. Both 50 mm square edges and the 100 mm calibration bar preserve their intended dimensions within 0.00002 mm digitally. The physical printer must still be checked with a ruler.

All tiled pages are A4. Their 190 x 250 mm frames, 180 x 240 mm advances and 10 mm overlaps cover the entire cut outlines. Shared registration marks are placed in every overlap, including regions without garment edges. Full-size and guide text remains within page bounds.

All three PDFs were rendered with Poppler. Every page was reviewed in contact sheets, with detailed inspection of the revised guide's measurement and cut-note layouts. Render review found no clipped cut outlines, overlapping guide text or missing pages. Intermediate review rasters were removed after inspection.

The `--no-tiles` guard refuses an output directory with an existing A4 set before writing artifact contents, so a changed size cannot leave an apparently current old A4 PDF. Use a new output directory for that option. Current PDF and JSON cut lists are authoritative if obsolete SVGs remain after disabling a component.

## Remaining physical work

Replace the illustrative measurements, print the calibration page, then sew a toile in similar-weight fabric. Confirm body mobility, front coverage, cuff passage, hood balance/view and hem clearance on the wearer. No claim of physical fit, optimized fabric yardage, device-load capacity or manufacturing certification is made.
