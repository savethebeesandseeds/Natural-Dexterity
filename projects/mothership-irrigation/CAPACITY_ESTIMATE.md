# Mothership Irrigation — first capacity estimate

Date: 12 October 2026.
Design basis: **C — soft capsule airship**, selected as the visual basis.
Status: concept sizing only; no development or component procurement requested.

**Estimate quality:** Fast preliminary estimates, intended for discussion. Component weights are rough planning allowances; the complete aircraft and its operating capacity have not been validated. A [three-page illustrated PDF](MOTHERSHIP_IRRIGATION_FAST_ESTIMATES.pdf) presents the estimates for reading and sharing.

An approximately **8 × 4 × 4 m envelope** has a preliminary allowance for **10–20 L of onboard water-like liquid**, after roughly **43–46 kg of dry equipment** and a 15% sizing allowance. A **10 × 5 × 5 m envelope** raises the same model to roughly **43–64 L**. These are static mass budgets, not established operating capacities: liquid discharge changes buoyancy and requires a trim strategy.

The image has no reliable dimensions. Its proportions are used to define a broad body with length approximately twice its diameter; the numbers below do not measure the generated picture.

## Assumptions and calculation

- Helium, dry air, 35 °C, pressure 101.325 kPa, helium and surrounding air at the same temperature; small envelope overpressure neglected.
- About 95% of the outer envelope volume contains helium and 5% contains ambient air, representing modest ballonet space. That internal air is already allowed for in the lift equation.
- Water and dilute mixtures are approximated as 1 kg/L. Actual capacity in litres is remaining payload mass divided by the actual liquid density.
- Two small custom drones carried on board, one working and one spare. Both are counted in the takeoff mass, with their batteries, cameras, guards, and nozzles.
- A lightweight service pod, local liquid storage and dispensing, approximately 1 kWh mothership battery for the 8 m case, and no ground supply hose in this particular estimate.
- Base envelope fabric: 200 g/m², plus 15% for seams and local reinforcements. Separate fins, fittings, valves, supports, and pressure-control hardware are counted separately.
- 15% of helium net lifting capacity is kept outside the planned dry-equipment-plus-liquid budget as a sizing allowance for mass growth and operating conditions. This is **not** an automatic buoyancy controller. Actual flight must be trimmed near neutral, potentially using some of that allowance as ballast, or supported by a tether/thrust.
- All unsourced individual part masses are engineering allowances, not weighed parts or a supplier bill of materials. The hardware budget includes preliminary pressure-control and docking allowances, but no resolved solution to large payload-release buoyancy changes.

Archimedes' principle and the ideal-gas relationship give:

```text
rho_air = P / (287.05 × T_K)
rho_helium = P / (2077.1 × T_K)

At 35 °C and 101.325 kPa:
rho_air ≈ 1.146 kg/m³
rho_helium ≈ 0.158 kg/m³
net helium lift ≈ 0.987 kg per m³ of helium

B = (rho_air - rho_helium) × 0.95 × V_outer
M_dry = M_envelope + M_other_hardware
M_liquid_budget = 0.85 × B - M_dry
litres ≈ max(0, M_liquid_budget) / (1 kg/L)
```

The helium's own mass has already been subtracted in B. Do not subtract it again as dry equipment. This mass accounting follows [NASA's buoyancy example](https://www.grc.nasa.gov/www/k-12/WindTunnel/Activities/ArchAnswer5.html) and [NASA's gas-density explanation](https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/gas-density/).

C's exact contour is undecided. Two idealized shapes bracket it:

1. **Prolate ellipsoid**, smoothly tapered: `V = π × L × D² / 6`.
2. **Spherocylinder**, cylindrical middle with hemispherical ends: `V = π × D² × (L - D) / 4 + π × D³ / 6`.

For L = 2D, the capsule contains 25% more gas than the ellipsoid with the same external dimensions, but also needs more fabric. Surface area is calculated separately for each shape:

```text
Ellipsoid: a = L/2, b = D/2, e = sqrt(1 - b²/a²)
A = 2πb² × [1 + a × asin(e)/(b × e)]

Spherocylinder: A = π × D × L

M_envelope = A × 0.200 kg/m² × 1.15
```

These are idealized geometry calculations, not a final seam pattern or structural model. A published [airship-envelope material study](https://pmc.ncbi.nlm.nih.gov/articles/PMC6165535/) reports a multilayer material at 200 g/m². That supports a mass reference; it does not establish this project's eventual material, strength, or lifetime.

## Size versus liquid mass budget

Dimensions refer to the main gas envelope, excluding fins, service pod, hose, and drone.

| Envelope length × width × height | Outer volume | Net helium lifting capacity B | Dry equipment excluding helium | Liquid budget after 15% allowance |
| --- | ---: | ---: | ---: | ---: |
| 5 × 2.5 × 2.5 m | 16–20 m³ | 15–19 kg | 28–30 kg | No capacity with this package |
| 6 × 3 × 3 m | 28–35 m³ | 27–33 kg | 32–34 kg | No capacity with this package |
| 7 × 3.5 × 3.5 m | 45–56 m³ | 42–53 kg | 37–40 kg | 0–5 L |
| 8 × 4 × 4 m | 67–84 m³ | 63–79 kg | 43–46 kg | 10–20 L |
| 10 × 5 × 5 m | 131–164 m³ | 123–153 kg | 61–66 kg | 43–64 L |
| 12 × 6 × 6 m | 226–283 m³ | 212–265 kg | 86–94 kg | 94–131 L |

Each range is the result of changing **shape**, from ellipsoid to full capsule, with the stated component allowances. It is not a statistical confidence interval. All sizes carry the two-drone mission package; a substantially stripped single-drone demonstrator could be smaller.

Other-hardware masses are explicit planning assumptions: 20.5 kg for 5 and 6 m; 22 kg for 7 m; 23.2 kg for 8 m; 30 kg for 10 m; and 42 kg for 12 m. Larger models allow heavier supports, propulsion, batteries, plumbing, and empty tanks; these allocations have not been structurally sized.

## Parts budget for an 8 × 4 × 4 m airship

| Component | Preliminary mass |
| --- | ---: |
| Envelope with seam allowance | 19.8–23.1 kg |
| Fins, attachment supports, valves and pressure-control allowance | 2.0 kg |
| Service pod and load frame | 3.0 kg |
| Thrusters, guards and motor controllers | 3.0 kg |
| Mothership battery (~1 kWh nominal) | 5.0 kg |
| Navigation, computer and obstacle sensing | 1.2 kg |
| Empty liquid containers | 1.5 kg |
| Pump, liquid valves and filters | 1.5 kg |
| Hose, fluid in hose, powered reel and connectors | 1.5 kg |
| Two drones including batteries, cameras, guards and nozzles | 3.0 kg |
| Docking, power distribution, wiring and miscellaneous | 1.5 kg |
| **Total dry equipment** | **43.0–46.3 kg** |

The envelope has approximately **86–101 m²** of surface. That is why its textile contributes approximately **20–23 kg**, even though the skin appears visually thin.

Published hardware provides useful scale checks:

- [Grepow/Tattu's battery table](https://www.grepow.com/uav-battery/tattu-6s-lipo-drone-battery.html) lists a 488.4 Wh pack at 2.46 kg. Two packs provide 976.8 Wh at 4.92 kg, supporting the roughly 5 kg battery allowance. Housing and power-distribution allowances appear elsewhere in the budget. This is an energy/mass reference, not a flight-duration calculation.
- [DJI's Mavic 3 Multispectral announcement](https://www.dji.com/newsroom/news/dji-launches-the-mavic-3-multispectral) gives a complete RGB/multispectral drone mass of 951 g. [FLIR's Hadron 640 series](https://www.flir.asia/products/hadron-640?segment=oem&vertical=lwir) lists a 56 g thermal/RGB module before integration. Those references make a 1.5 kg custom drone target worth examining, but do not validate the combined drone, extra sensors, guarding, or hose forces.
- [Xylem/Flojet's low-flow pump datasheet](https://www.xylem.com/siteassets/brand/flojet/resources/brochure/lf_plus_industrial_ss_f100-030.pdf) lists about half a kilogram per small pump (1.25 lb), so 1.5 kg for pumps, valves, and filters is a preliminary lightweight allocation. Flow, pressure, duty cycle, liquid compatibility, and multiple isolated liquid circuits remain undecided.
- As a reality check, [eBlimp's outdoor systems](https://www.eblimp.com/remote-control-airship-packages/outoor-rc-zeppelin-blimps.html) list a much narrower 6 m × 1.37 m body at about 5.7 m³, carrying a typical 2.2 kg payload. Length alone says little: C's broad diameter is what enables the larger mass budget.

The 1.5 kg hose/reel allocation assumes a roughly 15 m lightweight hose and includes fluid already inside it. For illustration, a 4 mm internal diameter hose contains about 0.19 L over 15 m. Tank liquid allowances are additional to that already counted hose fluid. A flying drone must also carry the suspended portion of the wet hose and resist tension and nozzle reaction; counting overall mass does not establish successful flight.

The **10–20 L is the sum across onboard tank contents**, including any water, dilute treatment solution, feed, and rinse liquid. It is not 10–20 L for every tank.

At that scale, the liquid alone occupies a cube roughly 22–27 cm on a side. A service enclosure around **0.9 × 0.45 × 0.4 m**, with separate compact external drone docks, is a packaging starting point; tanks, pumps, reel, battery, heat management, and structural clearances still need an actual layout.

## How sensitive is the estimate?

At 8 × 4 × 4 m, keeping all other assumptions unchanged:

| Fabric areal mass | Liquid budget, ellipsoid to capsule |
| --- | ---: |
| 150 g/m² | 15–26 L |
| 200 g/m² | 10–20 L |
| 300 g/m² | 1–9 L |

Fabric values other than 200 g/m² are sensitivity cases, not selected or validated products. Each additional kilogram of hardware, ballast, or stored equipment removes approximately one litre of water-like capacity.

Liquid capacities here assume dry fabric, calm conditions, and the stated density. Wind control, pressure, actual gas purity, environmental conditions, wet fabric, and part growth can reduce the result. [Allsopp Helikites' sizing notes](https://www.helikites.com/helikite-prices) similarly identify line mass, heat, humidity, and wet-envelope mass as deductions from usable aerostat lift.

## Liquid discharge and hovering

If the airship starts neutrally buoyant and dispenses 10 L, its weight falls by roughly 10 kg. Without another change, the upward force becomes roughly **98 N**. Dispensing 20 L produces about **196 N** of upward imbalance. Deploying a previously docked drone also changes the load distribution and support forces.

This is intrinsic to using a buoyant airship to dispense a consumable payload. Initial lifting capacity alone does not solve it. Options to examine are:

- A controlled ground tether/winch that reacts excess upward lift.
- Downward or vectorable thrust, sized for the empty-liquid condition, with its required power and battery mass.
- An engineered buoyancy/ballast system. Ordinary low-pressure ballonets are mainly pressure/trim devices; replacing 10–20 kg of discharged water with ambient-air ballast would require roughly 9–17 m³ of air at these conditions and substantial helium compression or gas-volume change. A little air bag or a small pressure adjustment cannot be assumed to solve that mass change.
- Frequent small-dose missions and ground refilling, combined with an actual means of descending and trimming between visits.

Venting helium loses lifting gas and is not assumed as the routine solution. The horizontal steering fans in the picture do not establish vertical control. No free-flight tank-emptying endurance is inferred from the 1 kWh battery allowance.

## Implication for the concept

**Use C as the visual basis, with an 8 m × 4 m envelope as a first mass-budget reference and approximately 10 L as the conservative onboard-liquid target.** A fuller capsule contour gives a central estimate closer to 20 L, but envelope material, propulsion, and buoyancy trim can consume that extra capacity.

A 10 m × 5 m envelope is a more plausible mass budget for roughly 40–60 L. Around 100 L, the same model is already in the approximately 12 m × 6 m envelope class.

Small onboard loads fit targeted plant care and spraying better than large-volume irrigation. For arithmetic only, 10 L divided into 0.1 L applications gives 100 applications; divided into 1 L applications gives 10. These are examples of capacity, not agronomic treatment rates.

The next sizing decision is the **buoyancy/trim architecture during dispensing**. It determines whether the attractive small mothership can actually use the liquid it can initially lift.
