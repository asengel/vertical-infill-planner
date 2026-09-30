"""
Example: vertical infill planning on the public Norne field model
==================================================================

Shows how to drive ``infill_planner`` from a full Eclipse simulation deck.
Norne (Equinor / OPM, Open Database License) is used only as a public test
case; nothing in the package is Norne-specific.

All inputs are downloaded from the OPM ``opm-data`` repository on first run
and cached in ``examples/norne/data/``. Results go to ``examples/norne/output/``.

Run from the repository root (after ``pip install -e .``):

    python examples/norne/run_norne.py

Workflow
--------
1. Field boundary: map outline of hydrocarbon columns (active cells whose top
   is above the oil-water contact of their EQUIL region).
2. Oil map: initial So from SWATINIT + EQUIL contacts (gas cap above the GOC),
   summed per grid column into *net oil pay* = sum(NTG x thickness) over cells
   with So >= SO_CUTOFF. This is the oil a vertical well would open to.
   Optionally use a simulated SOIL (e.g. end of history) instead - see SOIL_FILE.
3. Existing wells from the schedule (COMPDAT -> X/Y), role from the last
   WCONHIST (producer) / WCONINJE (injector) record; trajectories count for
   spacing.
4. New vertical wells on a hexagonal grid at WELL_SPACING, kept at least
   WELL_SPACING from every existing well *trajectory* (horizontal sections
   included).
5. One figure with a panel per spacing, drawn on the top-reservoir depth
   (structure map with contours).
6. Roles ("scored"):
   - eligibility: producer needs >= PRODUCER_OWC_STANDOFF from the OWC
     outline and >= MIN_NET_OIL_PAY; injector needs >= INJECTOR_OWC_STANDOFF
     and must not be in the gas cap;
   - count: INJECTOR_RATIO over the whole field (existing + new wells);
   - choice: injectors spread out and on the leanest pay (PAY_WEIGHT).

Edit the settings block below to change spacing, ratio, standoffs, ...
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# Use the installed package (`pip install -e .`); otherwise run straight from a clone.
if importlib.util.find_spec("infill_planner") is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from infill_planner import InfillConfig, InfillPlanner
from infill_planner.eclipse_io import (CornerPointGrid, column_property_map, initial_saturations,
                                       net_pay_maps, read_equil, read_schedule_wells, top_structure)

# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
OUT_DIR = HERE / "output"

WELL_SPACING = 750.0          # m, minimum distance between any two wells
INJECTOR_RATIO = "1:2"        # injector:producer target ...
RATIO_SCOPE = "field"         # ... over existing + new wells ("new" = new wells only)
LAYOUT = "hex"                # "hex", "square" or "poisson"
PRODUCER_OWC_STANDOFF = 300.0 # m inside the OWC outline for producers (coning / early water)
INJECTOR_OWC_STANDOFF = 100.0 # m inside the OWC outline for injectors (flank / peripheral)
PAY_WEIGHT = 0.5              # injector choice: 0 = spread only, 1 = leanest pay only
SO_CUTOFF = 0.5               # cell counts as oil pay if So >= this
MIN_NET_OIL_PAY = 10.0        # m, minimum net oil pay for a new producer
SOIL_FILE = None              # optional GRDECL with SOIL (e.g. exported end-of-history
                              # oil saturation); None = initial So from SWATINIT
SPACINGS = (750, 600, 500, 400)  # a plan + map is made for each spacing
DEPTH_RANGE = (2450.0, 2750.0)   # m TVD, colour range of the top-structure map
CONTOUR_INTERVAL = 25.0          # m, depth contour interval (labels every 2nd)
MIN_PART_AREA = 0.1e6         # m2, drop outline slivers smaller than this
SMOOTH = 50.0                 # m, smooths the stair-stepped grid outline

OPM_URL = "https://raw.githubusercontent.com/OPM/opm-data/master/norne/INCLUDE"
FILES = {
    "grid": ("IRAP_1005.GRDECL", "GRID/IRAP_1005.GRDECL"),
    "actnum": ("ACTNUM_0704.prop", "GRID/ACTNUM_0704.prop"),
    "schedule": ("BC0407_HIST01122006.SCH", "BC0407_HIST01122006.SCH"),
    "eqlnum": ("EQLNUM_0704.prop", "PETRO/EQLNUM_0704.prop"),
    "equil": ("E3.prop", "PETRO/E3.prop"),
    "ntg": ("NTG_0704.prop", "PETRO/NTG_0704.prop"),
    "swatinit": ("SWINITIAL.INC", "PETRO/SWINITIAL.INC"),
}

log = logging.getLogger("norne_infill")


def fetch_inputs() -> dict[str, Path]:
    """Return local paths of all inputs, downloading missing ones from OPM."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, (name, remote) in FILES.items():
        path = DATA_DIR / name
        if not path.exists():
            log.info("Downloading %s", name)
            urllib.request.urlretrieve(f"{OPM_URL}/{remote}", path)
        paths[key] = path
    return paths


def outline_to_table(geom) -> pd.DataFrame:
    """(Multi)Polygon -> vertex table with a Zone column (one zone per part)."""
    rows = []
    for n, p in enumerate(getattr(geom, "geoms", [geom]), start=1):
        for x, y in list(p.exterior.coords)[:-1]:
            rows.append({"Zone": f"Part-{n}", "X": round(x, 2), "Y": round(y, 2)})
    return pd.DataFrame(rows)


def make_config(spacing: float, prefix: str = "INF") -> InfillConfig:
    return InfillConfig(well_spacing=spacing, layout=LAYOUT, injector_ratio=INJECTOR_RATIO,
                        ratio_scope=RATIO_SCOPE, role_method="scored",
                        producer_standoff=PRODUCER_OWC_STANDOFF,
                        injector_standoff=INJECTOR_OWC_STANDOFF,
                        min_producer_quality=MIN_NET_OIL_PAY, pay_weight=PAY_WEIGHT,
                        new_well_prefix=prefix)


def plot_title(res, spacing: float) -> str:
    s = res.summary
    return (f"Spacing {spacing:g} m: {s['new_wells']} new wells "
            f"({s['new_injectors']} inj / {s['new_producers']} prod)\n"
            f"field injector fraction after infill: {s['field_injector_fraction_after']:.2f}")


def plot_all(results: dict, depth_map, path: Path) -> None:
    """One panel per spacing, shared depth colour bar and legend."""
    import matplotlib.pyplot as plt

    n = len(results)
    cols = 2 if n > 1 else 1
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.5 * cols, 8.2 * rows), squeeze=False,
                             layout="constrained")
    for ax, (sp, res) in zip(axes.ravel(), results.items()):
        res.plot(ax=ax, background=depth_map, cmap="Spectral", vmin=DEPTH_RANGE[0],
                 vmax=DEPTH_RANGE[1], contour_interval=CONTOUR_INTERVAL,
                 colorbar=False, legend=False, title=plot_title(res, sp))
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    # legend entries from all panels (a panel may lack e.g. new producers)
    entries: dict = {}
    for ax in axes.ravel()[:n]:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            entries.setdefault(lab, h)
    fig.legend(entries.values(), entries.keys(), loc="outside lower center", ncol=5,
               frameon=False, fontsize=10)
    first = next(iter(results.values()))
    cb = fig.colorbar(first._last_mappable, ax=axes.ravel().tolist(), shrink=0.4,
                      location="right", pad=0.01)
    cb.set_label(depth_map.name)
    cb.ax.invert_yaxis()  # shallow at the top
    fig.suptitle("Norne vertical infill plans on top-reservoir depth", fontsize=15)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    f = fetch_inputs()
    OUT_DIR.mkdir(exist_ok=True)

    # 1. grid, contacts, hydrocarbon outline ---------------------------------
    grid = CornerPointGrid.from_files(f["grid"], f["actnum"])
    eqlnum = grid.read_property(f["eqlnum"], "EQLNUM")
    equil = read_equil(f["equil"])
    owc = dict(zip(equil.region.astype(int), equil.owc))
    hc_cols = grid.hydrocarbon_columns(eqlnum, owc)
    boundary = grid.outline(hc_cols, min_part_area=MIN_PART_AREA)
    boundary = boundary.buffer(SMOOTH).buffer(-SMOOTH).simplify(SMOOTH / 2)
    log.info("Hydrocarbon outline %.2f km2; contacts:\n%s", boundary.area / 1e6, equil.to_string(index=False))

    # 2. oil and depth maps -------------------------------------------------------
    ntg = grid.read_property(f["ntg"], "NTG")
    swat = grid.read_property(f["swatinit"], "SWATINIT")
    so, sg = initial_saturations(grid, swat, eqlnum, equil)
    if SOIL_FILE:
        so = grid.read_property(SOIL_FILE, "SOIL")
        log.info("Using simulated SOIL from %s", SOIL_FILE)
    pay = net_pay_maps(grid, ntg, so, sg, cutoff=SO_CUTOFF)
    gas_dominated = (pay["net_gas_pay"] > pay["net_oil_pay"]) & (pay["net_oil_pay"] < MIN_NET_OIL_PAY)
    active_cols = grid.active_columns()
    # net oil pay drives the roles; the maps show top-reservoir depth
    oil_map = column_property_map(grid, pay["net_oil_pay"], active_cols, "net_oil_pay_m",
                                  injector_ok2d=~gas_dominated)
    depth_map = column_property_map(grid, top_structure(grid), active_cols,
                                    "Top reservoir depth (m TVD)")

    # 3. existing wells --------------------------------------------------------
    wells, paths = read_schedule_wells(f["schedule"], grid)
    wells["net_oil_pay_m"] = oil_map.sample(wells[["x", "y"]].to_numpy())[0]
    wells["top_depth_m"] = depth_map.sample(wells[["x", "y"]].to_numpy())[0]
    log.info("Existing wells: %d (%d producers, %d injectors)", len(wells),
             (wells.role == "producer").sum(), (wells.role == "injector").sum())

    # 4. one plan per spacing -----------------------------------------------------
    planner = InfillPlanner(boundary, wells, keep_outside_wells=True, well_paths=paths)
    results, rows = {}, []
    for sp in SPACINGS:
        res = planner.run(make_config(sp), oil_map)
        res.wells["top_depth_m"] = depth_map.sample(res.wells[["x", "y"]].to_numpy())[0]
        results[sp] = res
        s = res.summary
        rows.append({"spacing_m": sp, "new_wells": s["new_wells"], "new_injectors": s["new_injectors"],
                     "new_producers": s["new_producers"],
                     "field_inj_fraction_after": s["field_injector_fraction_after"],
                     "min_producer_net_oil_pay_m": round(s["new_producers_min_net_oil_pay_m"], 1),
                     "spacing_violations": s["spacing_violations"]})
    sens = pd.DataFrame(rows)
    results[SPACINGS[0]].print_summary()
    print("\nSpacing sensitivity\n" + sens.to_string(index=False))

    # 5. outputs -------------------------------------------------------------------
    sens.to_csv(OUT_DIR / "norne_spacing_sensitivity.csv", index=False)
    outline_to_table(boundary).to_csv(OUT_DIR / "norne_boundary.csv", index=False)
    (OUT_DIR / "norne_boundary.wkt").write_text(boundary.wkt)
    wells.to_csv(OUT_DIR / "norne_existing_wells.csv", index=False)
    paths.to_csv(OUT_DIR / "norne_well_paths.csv", index=False)
    pd.DataFrame({"X": oil_map.xy[:, 0].round(1), "Y": oil_map.xy[:, 1].round(1),
                  "net_oil_pay_m": oil_map.values.round(2),
                  "top_depth_m": depth_map.values.round(1)}) \
        .to_csv(OUT_DIR / "norne_maps.csv", index=False)

    with pd.ExcelWriter(OUT_DIR / "norne_infill_plans.xlsx") as xl:
        sens.to_excel(xl, sheet_name="sensitivity", index=False)
        for sp, res in results.items():
            res.new_wells.to_excel(xl, sheet_name=f"new_wells_{sp:g}m", index=False)
        wells.to_excel(xl, sheet_name="existing_wells", index=False)

    for sp, res in results.items():
        res.to_file(OUT_DIR / f"norne_infill_plan_{sp:g}m.csv")
    plot_all(results, depth_map, OUT_DIR / "norne_infill_plans.png")
    print(f"\nOutputs written to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
