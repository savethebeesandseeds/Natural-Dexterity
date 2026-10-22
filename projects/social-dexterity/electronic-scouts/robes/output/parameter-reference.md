# Measurement and option reference

All dimensions in millimeters. Defaults are illustrative. Use a wearer-specific JSON file.

| Key | Current value | Meaning |
| --- | --- | --- |
| `chest` | 680 | Chest circumference over the clothes worn beneath. |
| `hip` | 720 | Fullest hip/seat circumference over underclothes. |
| `waist` | 630 | Waist circumference at the chosen sash level. |
| `shoulder_width` | 310 | Back width between the anatomical shoulder points. |
| `neck_circumference` | 300 | Comfortably measured circumference at the neck base. |
| `head_circumference` | 540 | Around the fullest part of the head, above eyebrows. |
| `head_height` | 285 | Vertical height from high shoulder/neck level to crown. |
| `shoulder_to_waist` | 340 | High shoulder beside neck down to the chosen sash level. |
| `upper_arm` | 230 | Fullest upper-arm circumference with arm relaxed. |
| `hand_circumference` | 170 | Around knuckles with thumb tucked, for passage through cuff. |
| `wrist` | 140 | Wrist circumference; recorded for possible future long-sleeve adjustment. |
| `sleeve_length` | 180 | Outer sleeve: anatomical shoulder point to finished edge along arm. |
| `tunic_sleeve_length` | 230 | Inner sleeve: anatomical shoulder point to finished edge along arm. |
| `robe_length` | 750 | High shoulder beside neck to finished outer hem. Default is a knee-length idea. |
| `tunic_length` | 510 | High shoulder beside neck to finished inner hem. |
| `chest_ease` | 160 | Added outer chest circumference. Inner uses 40 mm less. |
| `hip_ease` | 160 | Added outer hip circumference. Inner uses 40 mm less. |
| `sleeve_ease` | 120 | Added outer upper-arm circumference at armhole. Inner uses 20 mm less. |
| `cuff_ease` | 70 | Added to max(upper arm, hand) for these loose cuffs. |
| `neck_ease` | 40 | Drafting addition to neck measurement for the open V-neck proportions. |
| `hood_ease` | 40 | Added independently to head/3 hood depth and shoulder-to-crown height. |
| `wrap_overlap` | 95 | Each front extends this far beyond centre front at sash/hem; total overlap is twice this. |
| `hem_flare` | 35 | Outward flare added at each front/back side hem. |
| `seam_allowance` | 10 | Joining seam allowance. Already included in CUT outlines. |
| `hem_allowance` | 25 | Total double-turn hem allowance; first turn 10 mm, second remainder. |
| `binding_width` | 10 | Finished double-fold bias binding width; cut strips four times this width. |
| `sash_width` | 50 | Finished width of the short fastening sash. |
| `sash_overlap` | 80 | End overlap for a low-profile hook-and-loop fastening. |
| `sash_ease` | 40 | Added waist circumference for sash over garment layers. |
| `tabard_width` | 85 | Finished width of each shoulder tabard. |
| `tabard_drop` | 110 | Distance tabard extends below sash level, front and back. |
| `include_hood` | True | Generate hood pieces. |
| `include_tunic` | True | Generate tunic pieces. |
| `include_tabards` | True | Generate tabards pieces. |
| `include_sash` | True | Generate sash pieces. |

## Formulas

- Body base circumference = max(chest + chest_ease, hip + hip_ease); inner subtracts 40 mm ease.
- Shoulder drop = quarter body circumference - half anatomical shoulder width; subtract drop from sleeve measurement.
- Arm opening on each body panel = half upper-sleeve circumference. Flat sleeve attachment = sum of both openings.
- Neck half-width = (neck + neck_ease)/6; back V depth = 0.065*(neck+ease); front V depth = 0.42*(neck+ease). These are design proportions, not an anthropometric sizing standard.
- Each hood base = measured back half-neck seam + measured front-neck seam; shoulder notch marks their junction.
- Hood upper depth = head circumference/3 + hood_ease; height = shoulder-to-crown height + hood_ease. Check hood vision and turning in a toile.
- Wrap overlap below sash = twice each front extension. Tabard length = 2*(shoulder_to_waist + tabard_drop).
- No allowance is added to bias-bound edges. Binding width is 4 times finished width, and supply includes joins plus 100 mm handling reserve.
- Tapered cuff and flared body hem extensions reflect adjacent seams at the final fold and again at the first 10 mm turn.
- Net cut area is piece area only; it is NOT fabric yardage and excludes layout waste, bias placement and shrinkage.

## Parameter notices

No notices for this configuration.
