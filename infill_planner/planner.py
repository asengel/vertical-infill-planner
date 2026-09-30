"""
infill_planner.planner
======================

Vertical infill well placement and injector/producer assignment inside a
field boundary.

New wells are vertical: each is a single map location (X, Y). Existing wells
may be vertical (one point) or deviated/horizontal (a trajectory, given as
``well_paths``); the spacing rule is then applied along the whole trajectory.

Workflow
--------
1. Load the field boundary (one or more polygons), the existing wells and,
   optionally, existing well trajectories and a reservoir-quality map.
2. Build the *developable area*: the boundary, optionally shrunk inward by a
   standoff distance.
3. Place new wells so that every well (new or existing, including wells just
   outside the boundary) is at least ``well_spacing`` apart:
      * ``hex``     - triangular/hexagonal lattice (densest regular packing)
      * ``square``  - square lattice
      * ``poisson`` - irregular Poisson-disk layout (Bridson's algorithm)
   Lattice layouts search over grid offset (and optionally rotation) to fit
   the most wells, then optionally fill leftover gaps near existing wells and
   along the boundary.
4. Assign roles to the new wells (optionally screened by a reservoir-quality
   map such as net oil pay or So, so producers never land in water/gas):
      * ``dispersed`` - any injector ratio; injectors are spread as evenly as
        possible (farthest-point sampling, seeded by existing injectors).
      * ``pattern``   - classic flood patterns on the lattice
        (5-spot, inverted 9-spot, line drive, 7-spot, inverted 7-spot).
5. Report QC metrics (incl. average distance to neighbours within a radius),
   export CSV/XLSX, and plot.

All distances are in the units of the input coordinates (use a projected CRS
such as UTM, *not* latitude/longitude).

Dependencies: numpy, pandas, scipy, shapely>=2.0, matplotlib (plotting only),
openpyxl (only for reading/writing .xlsx).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely import contains_xy, make_valid, prepare
from shapely import wkt as shapely_wkt
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

log = logging.getLogger("infill_planner")

Layout = Literal["hex", "square", "poisson"]
RoleMethod = Literal["dispersed", "pattern", "scored", "quality"]
RatioScope = Literal["new", "field"]

PRODUCER = "producer"
INJECTOR = "injector"
UNKNOWN = "unknown"

# Pattern name -> (lattice it needs, injector fraction it implies)
PATTERNS: dict[str, tuple[str, float]] = {
    "5-spot": ("square", 1 / 2),
    "inverted-9-spot": ("square", 1 / 4),
    "line-drive": ("square", 1 / 2),
    "7-spot": ("hex", 1 / 3),
    "inverted-7-spot": ("hex", 2 / 3),
}

_X_CANDIDATES = ("x", "surface x", "surface_x", "easting", "east", "utm x", "utm_x", "xcoord")
_Y_CANDIDATES = ("y", "surface y", "surface_y", "northing", "north", "utm y", "utm_y", "ycoord")
_NAME_CANDIDATES = ("well name", "well_name", "wellname", "well", "name", "uwi")
_ROLE_CANDIDATES = ("role", "type", "well type", "well_type", "status", "purpose")
_POLY_ID_CANDIDATES = ("zone", "polygon", "polygon_id", "part", "block", "id")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def parse_ratio(value: float | str) -> float:
    """Return the injector *fraction* from a ratio.

    Accepts a fraction (``0.33``), a percentage (``"33%"``) or an
    injector:producer ratio (``"1:2"`` -> 1/3).
    """
    if isinstance(value, str):
        v = value.strip()
        if ":" in v:
            inj, prod = (float(p) for p in v.split(":", 1))
            if inj < 0 or prod < 0 or inj + prod == 0:
                raise ValueError(f"Invalid injector:producer ratio {value!r}")
            return inj / (inj + prod)
        if v.endswith("%"):
            value = float(v[:-1]) / 100
        else:
            value = float(v)
    frac = float(value)
    if not 0.0 <= frac <= 1.0:
        raise ValueError(f"Injector fraction must be within [0, 1], got {frac}")
    return frac


@dataclass
class InfillConfig:
    """All planning parameters in one place.

    Attributes
    ----------
    well_spacing:
        Minimum distance between any two wells (new-new and new-existing).
    layout:
        ``"hex"``, ``"square"`` or ``"poisson"``.
    boundary_standoff:
        Minimum distance from new wells to the field boundary.
        ``None`` = half the well spacing, so each well's drainage area stays
        inside the field.
    grid_rotation_deg:
        Lattice rotation (counter-clockwise, degrees). ``None`` = search for
        the rotation that fits the most wells. Often set to the structural
        trend or the max horizontal stress direction.
    optimize_offset:
        Search lattice translations for the best fit.
    offset_trials:
        Offsets tried per lattice axis (total trials = offset_trials**2).
    rotation_step_deg:
        Step for the rotation search when ``grid_rotation_deg`` is None.
    fill_gaps:
        After lattice placement, add irregular wells where gaps remain.
    max_new_wells:
        Optional cap. Wells nearest to existing development are kept first.
    injector_ratio:
        Fraction (0.25), percentage ("25%") or ratio ("1:3").
    ratio_scope:
        ``"new"`` - ratio applies to new wells only.
        ``"field"`` - ratio applies to the whole field (existing injectors
        count towards the target).
    role_method:
        ``"dispersed"`` - ratio + even geometric spread of injectors.
        ``"pattern"``   - classic flood patterns on the lattice.
        ``"scored"``    - (recommended with reservoir data) three steps:
          1. eligibility: a location may be a producer only if it is at
             least ``producer_standoff`` inside the boundary and has
             quality >= ``min_producer_quality``; an injector only if it is
             ``injector_standoff`` inside and allowed by the quality map
             (e.g. not in a gas cap);
          2. count: the injector ratio (see ``ratio_scope``) sets how many
             injectors; surplus injector-only spots are dropped;
          3. choice: among spots eligible for both, injectors are picked
             greedily by score = (1 - pay_weight) * spread + pay_weight *
             low_quality, so injectors are spread out *and* sit on the
             leaner rock while the thickest pay is kept for producers.
        ``"quality"`` is an alias of ``"scored"``.
        For ``dispersed`` / ``pattern`` the same eligibility rules are applied
        afterwards (a non-eligible producer becomes an injector if allowed,
        otherwise the location is dropped, and vice versa).
    producer_standoff, injector_standoff:
        Role-specific minimum distance to the boundary (e.g. the oil-water
        contact outline). Producers usually need more (water coning, early
        breakthrough), injectors less. ``None`` = ``boundary_standoff``.
    min_producer_quality:
        Minimum quality-map value for a producer (e.g. 10 m net oil pay).
    pay_weight:
        0..1 weight of low quality vs. spatial spread when choosing
        injectors in ``"scored"`` (0 = spread only, 1 = pay only).
    pattern:
        Flood pattern name when ``role_method="pattern"`` (see ``PATTERNS``).
    neighbor_radius:
        Radius for the neighbour QC (average distance to wells within the
        radius). ``None`` = 1.5 x well_spacing.
    new_well_prefix:
        Prefix for generated well names.
    random_seed:
        Seed for reproducible Poisson/gap-fill layouts.
    """

    well_spacing: float = 750.0
    layout: Layout = "hex"
    boundary_standoff: float | None = None
    grid_rotation_deg: float | None = 0.0
    optimize_offset: bool = True
    offset_trials: int = 8
    rotation_step_deg: float = 5.0
    fill_gaps: bool = True
    max_new_wells: int | None = None
    injector_ratio: float | str = 0.33
    ratio_scope: RatioScope = "new"
    role_method: RoleMethod = "dispersed"
    pattern: str | None = None
    min_producer_quality: float | None = None
    producer_standoff: float | None = None
    injector_standoff: float | None = None
    pay_weight: float = 0.5
    neighbor_radius: float | None = None
    new_well_prefix: str = "NEW"
    random_seed: int | None = 42

    def __post_init__(self) -> None:
        if self.well_spacing <= 0:
            raise ValueError("well_spacing must be positive")
        if self.boundary_standoff is None:
            self.boundary_standoff = 0.5 * self.well_spacing
        if self.boundary_standoff < 0:
            raise ValueError("boundary_standoff cannot be negative")
        if self.layout not in ("hex", "square", "poisson"):
            raise ValueError(f"Unknown layout {self.layout!r}")
        if self.ratio_scope not in ("new", "field"):
            raise ValueError(f"Unknown ratio_scope {self.ratio_scope!r}")
        if self.role_method == "quality":
            self.role_method = "scored"
        if self.role_method not in ("dispersed", "pattern", "scored"):
            raise ValueError(f"Unknown role_method {self.role_method!r}")
        if self.offset_trials < 1:
            raise ValueError("offset_trials must be >= 1")
        if self.max_new_wells is not None and self.max_new_wells < 0:
            raise ValueError("max_new_wells cannot be negative")
        self.injector_ratio = parse_ratio(self.injector_ratio)
        if self.neighbor_radius is None:
            self.neighbor_radius = 1.5 * self.well_spacing
        elif self.neighbor_radius <= 0:
            raise ValueError("neighbor_radius must be positive")

        if self.producer_standoff is None:
            self.producer_standoff = self.boundary_standoff
        if self.injector_standoff is None:
            self.injector_standoff = self.boundary_standoff
        if min(self.producer_standoff, self.injector_standoff) < 0:
            raise ValueError("standoffs cannot be negative")
        if not 0.0 <= self.pay_weight <= 1.0:
            raise ValueError("pay_weight must be within [0, 1]")
        if self.role_method == "pattern":
            if self.pattern not in PATTERNS:
                raise ValueError(f"pattern must be one of {list(PATTERNS)}, got {self.pattern!r}")
            needed = PATTERNS[self.pattern][0]
            if self.layout != needed:
                raise ValueError(
                    f"Pattern {self.pattern!r} needs layout={needed!r} (got {self.layout!r})"
                )


# --------------------------------------------------------------------------- #
# Input / output
# --------------------------------------------------------------------------- #
def read_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV or Excel file into a DataFrame."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".csv", ".txt"):
        return pd.read_csv(path)
    if suffix in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported table format: {path.suffix}")


def _find_column(df: pd.DataFrame, candidates: Sequence[str], explicit: str | None,
                 required: bool = True) -> str | None:
    """Resolve a column by explicit name or by case-insensitive candidates."""
    if explicit is not None:
        if explicit not in df.columns:
            raise KeyError(f"Column {explicit!r} not found. Available: {list(df.columns)}")
        return explicit
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lookup:
            return lookup[cand]
    if required:
        raise KeyError(f"None of {candidates} found. Available: {list(df.columns)}")
    return None


def _clean_polygon(poly: BaseGeometry, label: str) -> BaseGeometry:
    if not poly.is_valid:
        log.warning("%s polygon is invalid (self-intersecting?); repairing with make_valid", label)
        poly = make_valid(poly)
        poly = unary_union([g for g in getattr(poly, "geoms", [poly])
                            if isinstance(g, (Polygon, MultiPolygon))])
    if poly.is_empty or poly.area == 0:
        raise ValueError(f"{label} polygon is empty or has zero area")
    return poly


def load_polygons(path: str | Path, x_col: str | None = None, y_col: str | None = None,
                  id_col: str | None = None) -> list[Polygon]:
    """Load polygons from CSV/XLSX vertices, GeoJSON or WKT.

    Tabular files hold one vertex per row, in drawing order. Several polygons
    can be stored in one file using an id column (``Zone``, ``Polygon``...).
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".geojson", ".json"):
        data = json.loads(path.read_text())
        feats = data.get("features", [data]) if isinstance(data, dict) else data
        geoms = [shape(f.get("geometry", f)) for f in feats]
    elif suffix == ".wkt":
        geoms = [shapely_wkt.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        df = read_table(path)
        xc = _find_column(df, _X_CANDIDATES, x_col)
        yc = _find_column(df, _Y_CANDIDATES, y_col)
        ic = _find_column(df, _POLY_ID_CANDIDATES, id_col, required=False)
        groups = df.groupby(ic, sort=False) if ic else [(None, df)]
        geoms = []
        for gid, g in groups:
            coords = g[[xc, yc]].to_numpy(dtype=float)
            if len(coords) < 3:
                raise ValueError(f"Polygon {gid!r} in {path.name} has fewer than 3 vertices")
            geoms.append(Polygon(coords))

    polys: list[Polygon] = []
    for g in geoms:
        g = _clean_polygon(g, path.name)
        polys.extend(g.geoms if isinstance(g, MultiPolygon) else [g])
    return polys


def normalize_role(value: object) -> str:
    """Map free-text well types (e.g. 'WI', 'Water Injector', 'OP') to a role."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return UNKNOWN
    v = str(value).strip().lower()
    if "inj" in v or v in ("wi", "gi", "wag", "i"):
        return INJECTOR
    if "prod" in v or v in ("op", "gp", "p", "oil", "gas"):
        return PRODUCER
    return UNKNOWN


def load_wells(path: str | Path, x_col: str | None = None, y_col: str | None = None,
               name_col: str | None = None, role_col: str | None = None) -> pd.DataFrame:
    """Load existing wells -> DataFrame[well_name, x, y, role]."""
    df = read_table(path)
    xc = _find_column(df, _X_CANDIDATES, x_col)
    yc = _find_column(df, _Y_CANDIDATES, y_col)
    nc = _find_column(df, _NAME_CANDIDATES, name_col, required=False)
    rc = _find_column(df, _ROLE_CANDIDATES, role_col, required=False)

    out = pd.DataFrame({
        "well_name": df[nc].astype(str) if nc else [f"EXIST-{i + 1:03d}" for i in range(len(df))],
        "x": pd.to_numeric(df[xc], errors="coerce"),
        "y": pd.to_numeric(df[yc], errors="coerce"),
        "role": df[rc].map(normalize_role) if rc else UNKNOWN,
    })
    bad = out[["x", "y"]].isna().any(axis=1)
    if bad.any():
        log.warning("Dropping %d well(s) with missing/non-numeric coordinates", int(bad.sum()))
        out = out[~bad]
    dup = out.duplicated(subset=["x", "y"])
    if dup.any():
        log.warning("Dropping %d duplicated well location(s)", int(dup.sum()))
        out = out[~dup]
    return out.reset_index(drop=True)


def load_well_paths(path: str | Path, name_col: str | None = None, x_col: str | None = None,
                    y_col: str | None = None, order_col: str | None = None) -> pd.DataFrame:
    """Load existing-well trajectories -> DataFrame[well_name, x, y, order].

    One row per point along a well path (e.g. survey stations or completion
    cells). Without an order column, the row order in the file is used.
    """
    df = read_table(path)
    nc = _find_column(df, _NAME_CANDIDATES, name_col)
    xc = _find_column(df, _X_CANDIDATES, x_col)
    yc = _find_column(df, _Y_CANDIDATES, y_col)
    oc = _find_column(df, ("order", "md", "measured depth", "station", "seq"), order_col, required=False)
    out = pd.DataFrame({"well_name": df[nc].astype(str),
                        "x": pd.to_numeric(df[xc], errors="coerce"),
                        "y": pd.to_numeric(df[yc], errors="coerce")})
    out["order"] = pd.to_numeric(df[oc], errors="coerce") if oc else out.groupby("well_name").cumcount()
    return out.dropna(subset=["x", "y"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def build_developable_area(boundary: BaseGeometry, standoff: float = 0.0) -> BaseGeometry:
    """Field boundary shrunk inward by ``standoff``."""
    area = boundary.buffer(-standoff) if standoff > 0 else boundary
    if area.is_empty:
        raise ValueError("Developable area is empty (standoff too large for the field)")
    return area


def densify_paths(paths: pd.DataFrame, step: float) -> pd.DataFrame:
    """Resample each well path (ordered x, y points) every ``step`` metres.

    ``paths`` needs columns well_name, x, y and optionally ``order``. Used so a
    long horizontal well blocks new locations along its whole length.
    """
    rows = []
    for name, g in paths.groupby("well_name", sort=False):
        if "order" in g:
            g = g.sort_values("order")
        pts = g[["x", "y"]].to_numpy(dtype=float)
        out = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            n = max(1, int(math.ceil(np.hypot(*(b - a)) / step)))
            t = np.linspace(0, 1, n + 1)[1:, None]
            out.extend(a + t * (b - a))
        arr = np.array(out)
        rows.append(pd.DataFrame({"well_name": name, "x": arr[:, 0], "y": arr[:, 1]}))
    if not rows:
        return pd.DataFrame(columns=["well_name", "x", "y"])
    return pd.concat(rows, ignore_index=True)


def neighbor_distance_stats(xy: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Average distance from each well to all other wells within ``radius``.

    Vectorised version of the legacy ``calculate_average_distances``: a KD-tree
    replaces the O(n^2) double loop. Returns (average_distance, neighbour_count);
    wells with no neighbour inside the radius get NaN (legacy returned 0, which
    reads like 'wells on top of each other').
    """
    n = len(xy)
    avg = np.full(n, np.nan)
    cnt = np.zeros(n, dtype=int)
    if n < 2:
        return avg, cnt
    pairs = cKDTree(xy).query_pairs(radius, output_type="ndarray")
    if len(pairs):
        d = np.hypot(*(xy[pairs[:, 0]] - xy[pairs[:, 1]]).T)
        idx = np.concatenate([pairs[:, 0], pairs[:, 1]])
        dd = np.concatenate([d, d])
        cnt = np.bincount(idx, minlength=n)
        sums = np.bincount(idx, weights=dd, minlength=n)
        with np.errstate(invalid="ignore", divide="ignore"):
            avg = np.where(cnt > 0, sums / cnt, np.nan)
    return avg, cnt


class _SpatialHash:
    """Uniform grid for fast 'is anything within r?' queries (cell size = r)."""

    def __init__(self, cell: float) -> None:
        self.cell = cell
        self.grid: dict[tuple[int, int], list[int]] = defaultdict(list)
        self.pts: list[tuple[float, float]] = []

    def _key(self, x: float, y: float) -> tuple[int, int]:
        return int(math.floor(x / self.cell)), int(math.floor(y / self.cell))

    def add(self, x: float, y: float) -> int:
        idx = len(self.pts)
        self.pts.append((x, y))
        self.grid[self._key(x, y)].append(idx)
        return idx

    def is_clear(self, x: float, y: float, r: float) -> bool:
        kx, ky = self._key(x, y)
        r2 = r * r
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for idx in self.grid.get((kx + dx, ky + dy), ()):
                    px, py = self.pts[idx]
                    if (px - x) ** 2 + (py - y) ** 2 < r2:
                        return False
        return True


# --------------------------------------------------------------------------- #
# Placement algorithms
# --------------------------------------------------------------------------- #
def _lattice_vectors(layout: str, s: float) -> tuple[np.ndarray, np.ndarray]:
    if layout == "hex":
        return np.array([s, 0.0]), np.array([s / 2, s * math.sqrt(3) / 2])
    return np.array([s, 0.0]), np.array([0.0, s])


def _lattice_candidates(area: BaseGeometry, tree: cKDTree | None, spacing: float, layout: str,
                        rotation_deg: float, offset_uv: tuple[float, float]
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Lattice nodes inside ``area`` and >= spacing from existing wells.

    Returns (xy[N, 2], ij[N, 2]); ij are integer lattice indices used for
    flood-pattern coloring.
    """
    a1, a2 = _lattice_vectors(layout, spacing)
    minx, miny, maxx, maxy = area.bounds
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    radius = math.hypot(maxx - minx, maxy - miny) / 2 + 2 * spacing

    m = math.ceil(radius / a2[1])
    n = math.ceil(radius / spacing) + m
    ii, jj = np.meshgrid(np.arange(-n, n + 1), np.arange(-m, m + 1))
    ii, jj = ii.ravel(), jj.ravel()

    u, v = offset_uv
    local = (ii + u)[:, None] * a1 + (jj + v)[:, None] * a2
    near = np.hypot(local[:, 0], local[:, 1]) <= radius
    local, ii, jj = local[near], ii[near], jj[near]

    th = math.radians(rotation_deg)
    c, s_ = math.cos(th), math.sin(th)
    x = local[:, 0] * c - local[:, 1] * s_ + cx
    y = local[:, 0] * s_ + local[:, 1] * c + cy

    inside = contains_xy(area, x, y)
    xy = np.column_stack([x[inside], y[inside]])
    ij = np.column_stack([ii[inside], jj[inside]])

    if tree is not None and len(xy):
        dist, _ = tree.query(xy)
        ok = dist >= spacing * (1 - 1e-9)
        xy, ij = xy[ok], ij[ok]
    return xy, ij


def place_lattice(area: BaseGeometry, existing_xy: np.ndarray, cfg: InfillConfig
                  ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Best-fitting lattice (searching offset and optionally rotation)."""
    prepare(area)
    tree = cKDTree(existing_xy) if len(existing_xy) else None

    period = 60.0 if cfg.layout == "hex" else 90.0
    if cfg.grid_rotation_deg is None:
        rotations = np.arange(0.0, period, cfg.rotation_step_deg)
    else:
        rotations = [float(cfg.grid_rotation_deg)]
    k = cfg.offset_trials if cfg.optimize_offset else 1
    offsets = [(i / k, j / k) for i in range(k) for j in range(k)]

    best = (np.empty((0, 2)), np.empty((0, 2), dtype=int))
    best_info = {"rotation_deg": float(rotations[0]), "offset_uv": (0.0, 0.0)}
    for rot in rotations:
        for off in offsets:
            xy, ij = _lattice_candidates(area, tree, cfg.well_spacing, cfg.layout, rot, off)
            if len(xy) > len(best[0]):
                best = (xy, ij)
                best_info = {"rotation_deg": float(rot), "offset_uv": off}
    log.info("Lattice (%s): %d wells, rotation %.1f deg, offset (%.3f, %.3f), %d trials",
             cfg.layout, len(best[0]), best_info["rotation_deg"], *best_info["offset_uv"],
             len(rotations) * len(offsets))
    return best[0], best[1], best_info


def place_poisson(area: BaseGeometry, fixed_xy: np.ndarray, spacing: float,
                  rng: np.random.Generator, k: int = 30, max_misses: int = 3000,
                  max_new: int | None = None) -> np.ndarray:
    """Bridson Poisson-disk sampling that respects already-placed wells.

    Grows outward from fixed wells near the area, then throws random darts so
    disconnected parts of the area (no nearby wells) are also filled.
    """
    prepare(area)
    grid = _SpatialHash(spacing)
    for x, y in fixed_xy:
        grid.add(float(x), float(y))
    new: list[tuple[float, float]] = []
    limit = max_new if max_new is not None else math.inf

    def grow(active: list[int]) -> None:
        while active and len(new) < limit:
            pos = int(rng.integers(len(active)))
            px, py = grid.pts[active[pos]]
            rad = spacing * np.sqrt(rng.uniform(1.0, 4.0, k))  # uniform in annulus [s, 2s]
            ang = rng.uniform(0.0, 2 * math.pi, k)
            cx, cy = px + rad * np.cos(ang), py + rad * np.sin(ang)
            inside = contains_xy(area, cx, cy)
            for x, y in zip(cx[inside], cy[inside]):
                if grid.is_clear(x, y, spacing):
                    active.append(grid.add(float(x), float(y)))
                    new.append((float(x), float(y)))
                    break
            else:  # no candidate accepted -> retire this seed
                active[pos] = active[-1]
                active.pop()

    # Seeds: fixed wells within reach of the developable area
    reach = area.buffer(2 * spacing)
    prepare(reach)
    if len(fixed_xy):
        near = contains_xy(reach, fixed_xy[:, 0], fixed_xy[:, 1])
        grow([int(i) for i in np.flatnonzero(near)])

    # Darts for regions not reached by growth
    minx, miny, maxx, maxy = area.bounds
    misses = 0
    while misses < max_misses and len(new) < limit:
        xs = rng.uniform(minx, maxx, 256)
        ys = rng.uniform(miny, maxy, 256)
        inside = contains_xy(area, xs, ys)
        for x, y in zip(xs[inside], ys[inside]):
            if grid.is_clear(x, y, spacing):
                idx = grid.add(float(x), float(y))
                new.append((float(x), float(y)))
                misses = 0
                grow([idx])
                break
            misses += 1
            if misses >= max_misses:
                break
    return np.array(new, dtype=float).reshape(-1, 2)


# --------------------------------------------------------------------------- #
# Role assignment
# --------------------------------------------------------------------------- #
def injector_target(n_new: int, frac: float, scope: str, n_ex_inj: int = 0, n_ex_prod: int = 0) -> int:
    """Number of new injectors for ``n_new`` new wells.

    scope "new":   round(frac * n_new)
    scope "field": round(frac * (existing + new)) - existing injectors,
                   clipped to [0, n_new]
    """
    if scope == "new":
        return int(round(frac * n_new))
    total = n_new + n_ex_inj + n_ex_prod
    return int(np.clip(round(frac * total) - n_ex_inj, 0, n_new))


def target_injector_count(n_new: int, existing_roles: np.ndarray, cfg: InfillConfig) -> int:
    """``injector_target`` from an array of existing roles."""
    return injector_target(n_new, float(cfg.injector_ratio), cfg.ratio_scope,
                           int(np.sum(existing_roles == INJECTOR)),
                           int(np.sum(existing_roles == PRODUCER)))


def assign_dispersed(new_xy: np.ndarray, n_injectors: int,
                     seed_injectors: np.ndarray) -> np.ndarray:
    """Pick ``n_injectors`` points spread as evenly as possible.

    Farthest-point sampling: each next injector is the candidate farthest from
    all injectors chosen so far (existing injectors act as initial seeds), so
    injectors never cluster and cover the field uniformly.
    """
    n = len(new_xy)
    roles = np.full(n, PRODUCER, dtype=object)
    if n == 0 or n_injectors <= 0:
        return roles
    n_injectors = min(n_injectors, n)

    if len(seed_injectors):
        d = cKDTree(seed_injectors).query(new_xy)[0]
    else:
        # No seed: start from the well nearest the centre of the new wells
        centre = new_xy.mean(axis=0)
        first = int(np.argmin(np.hypot(*(new_xy - centre).T)))
        roles[first] = INJECTOR
        d = np.hypot(*(new_xy - new_xy[first]).T)
        d[first] = -np.inf
        n_injectors -= 1

    for _ in range(n_injectors):
        pick = int(np.argmax(d))
        roles[pick] = INJECTOR
        d = np.minimum(d, np.hypot(*(new_xy - new_xy[pick]).T))
        d[roles == INJECTOR] = -np.inf
    return roles


def assign_pattern(ij: np.ndarray, pattern: str) -> np.ndarray:
    """Flood-pattern roles from integer lattice indices."""
    i, j = ij[:, 0], ij[:, 1]
    if pattern == "5-spot":
        inj = (i + j) % 2 == 0
    elif pattern == "inverted-9-spot":
        inj = (i % 2 == 0) & (j % 2 == 0)
    elif pattern == "line-drive":
        inj = j % 2 == 0
    elif pattern == "7-spot":            # 1 injector ringed by 6 producers
        inj = (i - j) % 3 == 0
    elif pattern == "inverted-7-spot":   # 1 producer ringed by 6 injectors
        inj = (i - j) % 3 != 0
    else:
        raise ValueError(f"Unknown pattern {pattern!r}")
    return np.where(inj, INJECTOR, PRODUCER).astype(object)


# --------------------------------------------------------------------------- #
# Reservoir-quality map
# --------------------------------------------------------------------------- #
def _area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


class PropertyMap:
    """Map-view property (e.g. net oil pay, So) sampled by nearest map node.

    Parameters
    ----------
    xy : (N, 2) node coordinates (e.g. grid-column centres or a gridded map)
    values : (N,) property values
    name : label used in outputs, e.g. "net_oil_pay_m"
    injector_ok : optional (N,) bool - where injectors may go (e.g. False in
        a gas cap). Default: everywhere.
    max_distance : points farther than this from any node get NaN
    polygons : optional list of (M, 2) arrays (cell outlines) for plotting
    """

    def __init__(self, xy: np.ndarray, values: np.ndarray, name: str = "quality",
                 injector_ok: np.ndarray | None = None, max_distance: float = np.inf,
                 polygons: list[np.ndarray] | None = None) -> None:
        self.xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        self.values = np.asarray(values, dtype=float)
        self.name = name
        self.injector_ok = (np.ones(len(self.values), dtype=bool) if injector_ok is None
                            else np.asarray(injector_ok, dtype=bool))
        self.max_distance = max_distance
        self.polygons = polygons
        self._tree = cKDTree(self.xy)

    @classmethod
    def from_table(cls, path: str | Path, value_col: str | None = None,
                   max_distance: float = np.inf) -> "PropertyMap":
        """Scattered map from CSV/XLSX with X, Y and one value column."""
        df = read_table(path)
        xc = _find_column(df, _X_CANDIDATES, None)
        yc = _find_column(df, _Y_CANDIDATES, None)
        if value_col is None:
            others = [c for c in df.columns if c not in (xc, yc)]
            if not others:
                raise KeyError("No value column in quality map")
            value_col = others[0]
        return cls(df[[xc, yc]].to_numpy(float), df[value_col].to_numpy(float),
                   name=str(value_col), max_distance=max_distance)

    def sample(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(value, injector_ok) at each point."""
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        if not len(xy):
            return np.zeros(0), np.zeros(0, dtype=bool)
        d, idx = self._tree.query(xy)
        far = d > self.max_distance
        val = self.values[idx].copy()
        val[far] = np.nan
        ok = self.injector_ok[idx] & ~far
        return val, ok


def _greedy_injectors(xy: np.ndarray, k: int, seeds: np.ndarray, quality: np.ndarray,
                      spacing: float, pay_weight: float) -> np.ndarray:
    """Pick ``k`` injector indices by spread + low quality (greedy).

    spread  = distance to the nearest injector (existing or already picked),
              normalised by 2 x spacing and capped at 1;
    lean    = 1 - quality / P90(quality), clipped to [0, 1] (unknown -> 0.5);
    score   = (1 - pay_weight) * spread + pay_weight * lean.
    """
    n = len(xy)
    if k <= 0 or n == 0:
        return np.zeros(0, dtype=int)
    k = min(k, n)
    seeds = np.asarray(seeds, dtype=float).reshape(-1, 2)
    d = cKDTree(seeds).query(xy)[0] if len(seeds) else np.full(n, np.inf)
    q = np.asarray(quality, dtype=float)
    finite = q[np.isfinite(q)]
    qref = np.percentile(finite, 90) if len(finite) and np.percentile(finite, 90) > 0 else 1.0
    lean = np.where(np.isfinite(q), 1.0 - np.clip(q / qref, 0.0, 1.0), 0.5)
    chosen: list[int] = []
    for _ in range(k):
        spread = np.clip(d / (2 * spacing), 0.0, 1.0)
        score = (1 - pay_weight) * spread + pay_weight * lean
        score[chosen] = -np.inf
        pick = int(np.argmax(score))
        chosen.append(pick)
        d = np.minimum(d, np.hypot(*(xy - xy[pick]).T))
    return np.array(chosen, dtype=int)


def assign_scored(new_xy: np.ndarray, can_prod: np.ndarray, can_inj: np.ndarray,
                  quality: np.ndarray, frac: float, scope: str, n_ex_inj: int, n_ex_prod: int,
                  seed_injectors: np.ndarray, spacing: float, pay_weight: float
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Eligibility -> count -> choice. Returns (roles, keep_mask).

    * producer-only spots are always kept as producers;
    * injector-only spots are kept only as far as the ratio needs them;
    * the remaining injectors come from spots eligible for both roles.
    """
    n = len(new_xy)
    roles = np.full(n, PRODUCER, dtype=object)
    prod_only = can_prod & ~can_inj
    inj_only = can_inj & ~can_prod
    both = can_prod & can_inj
    p0, n_io, n_both = int(prod_only.sum() + both.sum()), int(inj_only.sum()), int(both.sum())

    # largest number k of injector-only spots the ratio can absorb
    k = max(kk for kk in range(n_io + 1)
            if injector_target(p0 + kk, frac, scope, n_ex_inj, n_ex_prod) >= kk)
    target = injector_target(p0 + k, frac, scope, n_ex_inj, n_ex_prod)
    from_both = min(target - k, n_both)
    if target - k > n_both:
        log.warning("Injector target %d not reachable: only %d suitable locations", target, k + n_both)

    keep = prod_only | both
    seeds = np.asarray(seed_injectors, dtype=float).reshape(-1, 2)
    idx = np.flatnonzero(inj_only)
    pick = idx[_greedy_injectors(new_xy[idx], k, seeds, quality[idx], spacing, pay_weight)]
    roles[pick] = INJECTOR
    keep[pick] = True
    if (n_io - k) > 0:
        log.info("Dropped %d injector-only location(s) not needed for the ratio", n_io - k)

    idx = np.flatnonzero(both)
    seeds = np.vstack([seeds, new_xy[pick]])
    pick2 = idx[_greedy_injectors(new_xy[idx], from_both, seeds, quality[idx], spacing, pay_weight)]
    roles[pick2] = INJECTOR
    return roles, keep


# --------------------------------------------------------------------------- #
# Planner
# --------------------------------------------------------------------------- #
@dataclass
class InfillResult:
    wells: pd.DataFrame
    boundary: BaseGeometry
    developable_area: BaseGeometry
    config: InfillConfig
    summary: dict = field(default_factory=dict)
    well_paths: pd.DataFrame | None = None  # existing trajectories (plotting)
    quality_map: "PropertyMap | None" = None
    producer_area: BaseGeometry | None = None
    injector_area: BaseGeometry | None = None

    @property
    def new_wells(self) -> pd.DataFrame:
        return self.wells[self.wells["status"] == "new"].reset_index(drop=True)

    def to_file(self, path: str | Path, new_only: bool = False) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.new_wells if new_only else self.wells
        if path.suffix.lower() in (".xlsx", ".xls"):
            with pd.ExcelWriter(path) as xl:
                df.to_excel(xl, sheet_name="wells", index=False)
                pd.DataFrame(list(self.summary.items()), columns=["metric", "value"]) \
                    .to_excel(xl, sheet_name="summary", index=False)
        else:
            df.to_csv(path, index=False)
        log.info("Wrote %s", path)
        return path

    def print_summary(self) -> None:
        width = max(len(k) for k in self.summary)
        print("\nInfill plan summary")
        print("-" * (width + 16))
        for k, v in self.summary.items():
            print(f"{k:<{width}} : {v:,.2f}" if isinstance(v, float) else f"{k:<{width}} : {v}")

    @staticmethod
    def _draw_contours(ax, bg: "PropertyMap", interval: float) -> None:
        """Labelled contours on the map nodes; triangles spanning gaps between
        separate blocks (long edges) are masked out."""
        import matplotlib.tri as mtri

        ok = np.isfinite(bg.values)
        xy, v = bg.xy[ok], bg.values[ok]
        if len(v) < 3:
            return
        tri = mtri.Triangulation(xy[:, 0], xy[:, 1])
        cell = float(np.sqrt(np.median([abs(_area(p)) for p in bg.polygons]))) if bg.polygons else 100.0
        t = tri.triangles
        edges = np.stack([np.hypot(*(xy[t[:, a]] - xy[t[:, b]]).T)
                          for a, b in ((0, 1), (1, 2), (2, 0))], axis=1)
        tri.set_mask(edges.max(axis=1) > 2.0 * cell)
        levels = np.arange(np.floor(v.min() / interval) * interval, v.max() + interval, interval)
        cs = ax.tricontour(tri, v, levels=levels, colors="#222222", linewidths=0.6, alpha=0.8, zorder=1)
        ax.clabel(cs, levels=levels[::2], fontsize=6.5, fmt="%.0f", inline=True)

    def plot(self, path: str | Path | None = None, show: bool = False,
             title: str | None = None, label_existing: bool = False,
             figsize: tuple[float, float] = (11, 9), ax=None,
             background: "PropertyMap | None" = None, cmap: str | None = None,
             vmin: float | None = None, vmax: float | None = None,
             colorbar: bool = True, legend: bool = True,
             contour_interval: float | None = None):
        """Map of the plan.

        background : map drawn under the wells (default: the quality map used
            for roles, if it has cell polygons), e.g. an initial Sw map.
        ax : draw into an existing matplotlib axis (for multi-panel figures);
            then ``path`` / ``show`` are ignored and the caller saves the figure.
        contour_interval : draw labelled contours of the background map at this
            interval (e.g. 25 m for a depth map).
        """
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection

        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        def _polys(g):
            return list(g.geoms) if hasattr(g, "geoms") else [g]

        bg = background if background is not None else self.quality_map
        self._last_mappable = None
        if bg is not None and bg.polygons is not None:
            pc = PolyCollection(bg.polygons, array=bg.values, cmap=cmap or "YlOrBr",
                                edgecolors="none", zorder=0, alpha=0.9)
            pc.set_clim(vmin, vmax)
            ax.add_collection(pc)
            self._last_mappable = pc
            if colorbar:
                cb = fig.colorbar(pc, ax=ax, shrink=0.5, pad=0.07, location="bottom", aspect=35)
                cb.set_label(bg.name)
            if contour_interval:
                self._draw_contours(ax, bg, contour_interval)
        else:
            for p in _polys(self.developable_area):
                if isinstance(p, Polygon):
                    ax.fill(*p.exterior.xy, color="#f3efe3", zorder=0)
                    for ring in p.interiors:
                        ax.fill(*ring.xy, color="white", zorder=0)
        for p in _polys(self.boundary):
            ax.plot(*p.exterior.xy, color="black", lw=1.6, zorder=1)
        for geom, col, lab in ((self.producer_area, "#0b5d1e", "Producer standoff"),
                               (self.injector_area, "#8e44ad", "Injector standoff")):
            if geom is None or geom.is_empty or geom.equals(self.boundary):
                continue
            for i, p in enumerate(_polys(geom)):
                if isinstance(p, Polygon):
                    ax.plot(*p.exterior.xy, color=col, lw=1.4, ls="--", zorder=1,
                            label=lab if i == 0 else None)

        styles = {
            ("existing", PRODUCER): dict(marker="s", c="#1e8449", label="Existing producer"),
            ("existing", INJECTOR): dict(marker="s", c="#00bcd4", label="Existing injector"),
            ("existing", UNKNOWN): dict(marker="s", c="#7f8c8d", label="Existing (other)"),
            ("new", PRODUCER): dict(marker="o", c="#f1c40f", label="New producer"),
            ("new", INJECTOR): dict(marker="v", c="#e74c3c", label="New injector"),
        }
        if self.well_paths is not None and len(self.well_paths):
            role_of = dict(zip(self.wells.well_name, self.wells.role))
            colors = {PRODUCER: "#1e8449", INJECTOR: "#00bcd4"}
            for name, g in self.well_paths.groupby("well_name", sort=False):
                g = g.sort_values("order") if "order" in g else g
                if len(g) >= 3:  # 3-point moving average removes grid stair-steps
                    g = g[["x", "y"]].rolling(3, center=True, min_periods=1).mean()
                ax.plot(g.x, g.y, color=colors.get(role_of.get(name), "#7f8c8d"),
                        lw=2.2, alpha=0.85, zorder=2, solid_capstyle="round")
            ax.plot([], [], color="grey", lw=2.2, label="Existing well path")
        for (status, role), st in styles.items():
            sub = self.wells[(self.wells.status == status) & (self.wells.role == role)]
            if len(sub):
                ax.scatter(sub.x, sub.y, s=70 if status == "new" else 55,
                           edgecolors="black", linewidths=0.8, zorder=3, **st)
        if label_existing:
            for _, w in self.wells[self.wells.status == "existing"].iterrows():
                ax.annotate(w.well_name, (w.x, w.y), xytext=(4, 3), textcoords="offset points",
                            fontsize=6.5, color="#222222", zorder=4)

        ax.set_aspect("equal")
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.set_xlabel("X (Easting)")
        ax.set_ylabel("Y (Northing)")
        s = self.summary
        ax.set_title(title or (
            f"Infill plan: {s['new_wells']} new wells ({s['new_injectors']} inj / "
            f"{s['new_producers']} prod) | spacing {self.config.well_spacing:g} | "
            f"{self.config.layout} layout"))
        if legend:
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
        ax.grid(alpha=0.25)
        if own_fig:
            fig.tight_layout()
            if path:
                fig.savefig(path, dpi=150, bbox_inches="tight")
                log.info("Saved plot %s", path)
            if show:
                plt.show()
        return fig, ax

class InfillPlanner:
    """Plan infill wells for a field.

    Example
    -------
    >>> planner = InfillPlanner(boundary, existing_wells)
    >>> result = planner.run(InfillConfig(well_spacing=750, injector_ratio="1:2"))
    >>> result.to_file("infill_plan.csv"); result.plot("infill_plan.png")
    """

    def __init__(self, boundary: BaseGeometry | Sequence[BaseGeometry],
                 existing_wells: pd.DataFrame | None = None,
                 keep_outside_wells: bool = False,
                 well_paths: pd.DataFrame | None = None,
                 path_step: float = 25.0) -> None:
        """
        Parameters
        ----------
        boundary : field outline polygon(s)
        existing_wells : well_name, x, y, role (one map point per well)
        keep_outside_wells : keep existing wells outside the boundary
        well_paths : optional well_name, x, y[, order] trajectory points of
            existing (deviated / horizontal) wells. When given, the spacing
            rule is applied to the whole trajectory, not just the well point.
        path_step : resampling interval along well paths (m)
        """
        if not isinstance(boundary, BaseGeometry):
            boundary = unary_union(list(boundary))
        self.boundary = _clean_polygon(boundary, "Boundary")

        if existing_wells is None or len(existing_wells) == 0:
            existing_wells = pd.DataFrame(columns=["well_name", "x", "y", "role"])
        missing = {"x", "y"} - set(existing_wells.columns)
        if missing:
            raise KeyError(f"existing_wells is missing columns {missing}")
        ew = existing_wells.copy()
        if "well_name" not in ew:
            ew["well_name"] = [f"EXIST-{i + 1:03d}" for i in range(len(ew))]
        ew["role"] = ew["role"].map(normalize_role) if "role" in ew else UNKNOWN
        ew = ew[["well_name", "x", "y", "role"]].astype({"x": float, "y": float})

        # Existing wells outside the boundary are dropped by default
        inside = contains_xy(self.boundary, ew.x.to_numpy(), ew.y.to_numpy()) if len(ew) \
            else np.zeros(0, dtype=bool)
        if (~inside).any() and not keep_outside_wells:
            log.warning("Ignoring %d existing well(s) outside the boundary: %s",
                        int((~inside).sum()), ", ".join(ew.loc[~inside, "well_name"]))
            ew = ew[inside]
        self.existing = ew.reset_index(drop=True)

        self.well_paths = None
        if well_paths is not None and len(well_paths):
            wp = well_paths[well_paths.well_name.isin(self.existing.well_name)].copy()
            self.well_paths = wp.reset_index(drop=True)
            self._path_pts = densify_paths(wp, path_step)
        else:
            self._path_pts = pd.DataFrame(columns=["well_name", "x", "y"])

    def _constraint_points(self) -> tuple[np.ndarray, np.ndarray]:
        """All points new wells must keep clear of, and which are injector points."""
        pts = pd.concat([self.existing[["well_name", "x", "y"]], self._path_pts],
                        ignore_index=True) if len(self._path_pts) else self.existing[["well_name", "x", "y"]]
        inj_names = set(self.existing.loc[self.existing.role == INJECTOR, "well_name"])
        xy = pts[["x", "y"]].to_numpy(dtype=float).reshape(-1, 2)
        return xy, pts.well_name.isin(inj_names).to_numpy()

    def run(self, cfg: InfillConfig, quality_map: PropertyMap | None = None) -> InfillResult:
        """Plan new wells. ``quality_map`` enables quality-based roles/screening."""
        rng = np.random.default_rng(cfg.random_seed)
        prod_area = self.boundary.buffer(-cfg.producer_standoff) if cfg.producer_standoff else self.boundary
        inj_area = self.boundary.buffer(-cfg.injector_standoff) if cfg.injector_standoff else self.boundary
        area = unary_union([prod_area, inj_area])
        if area.is_empty:
            raise ValueError("Developable area is empty (standoffs too large for the field)")
        ex_roles = self.existing["role"].to_numpy(dtype=object)
        con_xy, con_is_inj = self._constraint_points()

        # ---- 1. placement -------------------------------------------------
        lattice_info: dict = {}
        if cfg.layout in ("hex", "square"):
            new_xy, ij, lattice_info = place_lattice(area, con_xy, cfg)
            placement = np.full(len(new_xy), "lattice", dtype=object)
            if cfg.fill_gaps:
                fixed = np.vstack([con_xy, new_xy]) if len(con_xy) else new_xy
                gap = place_poisson(area, fixed, cfg.well_spacing, rng)
                if len(gap):
                    log.info("Gap fill added %d off-lattice wells", len(gap))
                    new_xy = np.vstack([new_xy, gap])
                    ij = np.vstack([ij, np.full((len(gap), 2), np.iinfo(np.int64).min)])
                    placement = np.concatenate([placement, np.full(len(gap), "gap_fill", dtype=object)])
        else:
            new_xy = place_poisson(area, con_xy, cfg.well_spacing, rng)
            ij = np.full((len(new_xy), 2), np.iinfo(np.int64).min)
            placement = np.full(len(new_xy), "poisson", dtype=object)

        # ---- 2. optional cap: keep wells closest to existing development ----
        if cfg.max_new_wells is not None and len(new_xy) > cfg.max_new_wells:
            ref = con_xy if len(con_xy) else np.atleast_2d(np.array(area.centroid.coords[0]))
            order = np.argsort(cKDTree(ref).query(new_xy)[0], kind="stable")[: cfg.max_new_wells]
            new_xy, ij, placement = new_xy[order], ij[order], placement[order]
            log.info("Capped to %d new wells (nearest to existing development)", cfg.max_new_wells)

        # ---- 3. roles ------------------------------------------------------
        if cfg.role_method == "pattern":
            roles = np.full(len(new_xy), PRODUCER, dtype=object)
            on_lattice = placement == "lattice"
            roles[on_lattice] = assign_pattern(ij[on_lattice], cfg.pattern)
            if (~on_lattice).any():
                log.info("%d off-lattice gap-fill wells set to producer", int((~on_lattice).sum()))
        elif cfg.role_method == "dispersed":
            n_inj = target_injector_count(len(new_xy), ex_roles, cfg)
            roles = assign_dispersed(new_xy, n_inj, con_xy[con_is_inj])
        else:
            roles = np.full(len(new_xy), PRODUCER, dtype=object)  # set below

        # eligibility per role: standoffs + quality map
        quality = np.full(len(new_xy), np.nan)
        can_prod = contains_xy(prod_area, new_xy[:, 0], new_xy[:, 1]) if len(new_xy) else np.zeros(0, bool)
        can_inj = contains_xy(inj_area, new_xy[:, 0], new_xy[:, 1]) if len(new_xy) else np.zeros(0, bool)
        if quality_map is not None and len(new_xy):
            quality, inj_ok = quality_map.sample(new_xy)
            can_inj &= inj_ok
            if cfg.min_producer_quality is not None:
                can_prod &= np.nan_to_num(quality, nan=-np.inf) >= cfg.min_producer_quality

        if cfg.role_method == "scored":
            roles, keep = assign_scored(
                new_xy, can_prod, can_inj, quality, float(cfg.injector_ratio), cfg.ratio_scope,
                int(np.sum(ex_roles == INJECTOR)), int(np.sum(ex_roles == PRODUCER)),
                con_xy[con_is_inj], cfg.well_spacing, cfg.pay_weight)
        else:  # screen dispersed / pattern roles with the same eligibility rules
            bad_p = (roles == PRODUCER) & ~can_prod
            bad_i = (roles == INJECTOR) & ~can_inj
            roles[bad_p & can_inj] = INJECTOR
            roles[bad_i & can_prod] = PRODUCER
            keep = ~((bad_p & ~can_inj) | (bad_i & ~can_prod))
        if (~keep).any():
            log.info("Dropped %d location(s) in total (not eligible / not needed)", int((~keep).sum()))
        new_xy, roles, placement, quality = new_xy[keep], roles[keep], placement[keep], quality[keep]

        # ---- 4. assemble ---------------------------------------------------
        new_df = pd.DataFrame({
            "well_name": [f"{cfg.new_well_prefix}-{i + 1:03d}" for i in range(len(new_xy))],
            "x": new_xy[:, 0] if len(new_xy) else [],
            "y": new_xy[:, 1] if len(new_xy) else [],
            "role": roles,
            "status": "new",
            "placement": placement,
            "quality": quality,
        })
        old_df = self.existing.assign(status="existing", placement="existing")
        parts = [df for df in (old_df, new_df) if len(df)]
        wells = pd.concat(parts, ignore_index=True) if parts else new_df
        if "quality" not in wells:
            wells["quality"] = np.nan
        wells = wells[["well_name", "status", "role", "placement", "x", "y", "quality"]]
        wells = wells.astype({"x": float, "y": float})

        all_xy = wells[["x", "y"]].to_numpy(dtype=float)
        if len(all_xy) > 1:
            d, _ = cKDTree(all_xy).query(all_xy, k=2)
            wells["nn_distance"] = d[:, 1]
        else:
            wells["nn_distance"] = np.nan
        # New wells: distance to the nearest *trajectory* point of existing wells too
        is_new = (wells.status == "new").to_numpy()
        if is_new.any() and len(self._path_pts):
            d_path, _ = cKDTree(con_xy).query(all_xy[is_new])
            wells.loc[is_new, "nn_distance"] = np.fmin(wells.loc[is_new, "nn_distance"], d_path)
        r = cfg.neighbor_radius
        avg, cnt = neighbor_distance_stats(all_xy, r)
        wells[f"avg_dist_within_{r:g}"] = avg
        wells[f"n_wells_within_{r:g}"] = cnt
        wells["inside_boundary"] = contains_xy(self.boundary, wells.x.to_numpy(), wells.y.to_numpy())

        if quality_map is not None:
            wells["quality"] = quality_map.sample(wells[["x", "y"]].to_numpy())[0]
            wells = wells.rename(columns={"quality": quality_map.name})
        else:
            wells = wells.drop(columns="quality")
        result = InfillResult(wells, self.boundary, area, cfg, well_paths=self.well_paths,
                              quality_map=quality_map, producer_area=prod_area,
                              injector_area=inj_area)
        result.summary = self._summarize(result, lattice_info)
        return result

    @staticmethod
    def _summarize(res: InfillResult, lattice_info: dict) -> dict:
        w, cfg = res.wells, res.config
        new = w[w.status == "new"]
        n_new, n_inj = len(new), int((new.role == INJECTOR).sum())
        known = w[w.role != UNKNOWN]
        inj_all = w.loc[w.role == INJECTOR, ["x", "y"]].to_numpy(dtype=float)
        prod_new = new.loc[new.role == PRODUCER, ["x", "y"]].to_numpy(dtype=float)

        support_r = 1.5 * cfg.well_spacing
        if len(inj_all) and len(prod_new):
            counts = np.array([len(c) for c in cKDTree(inj_all).query_ball_point(prod_new, support_r)])
            supported, mean_inj = float((counts > 0).mean() * 100), float(counts.mean())
        else:
            supported, mean_inj = 0.0, 0.0

        s = {
            "existing_wells": int((w.status == "existing").sum()),
            "new_wells": n_new,
            "new_producers": n_new - n_inj,
            "new_injectors": n_inj,
            "new_injector_fraction": round(n_inj / n_new, 3) if n_new else 0.0,
            "existing_producers": int(((w.status == "existing") & (w.role == PRODUCER)).sum()),
            "existing_injectors": int(((w.status == "existing") & (w.role == INJECTOR)).sum()),
            "target_injector_fraction": round(float(cfg.injector_ratio), 3),
            "field_injector_fraction_after": round(float((known.role == INJECTOR).mean()), 3)
            if len(known) else 0.0,
            "min_spacing_new_wells": float(new.nn_distance.min()) if n_new else float("nan"),
            "mean_spacing_new_wells": float(new.nn_distance.mean()) if n_new else float("nan"),
            "spacing_violations": int((new.nn_distance < cfg.well_spacing * (1 - 1e-6)).sum()),
            "developable_area": float(res.developable_area.area),
            "area_per_new_well": float(res.developable_area.area / n_new) if n_new else float("nan"),
            f"new_producers_with_injector_within_{support_r:g} (%)": round(supported, 1),
            "mean_injectors_per_new_producer": round(mean_inj, 2),
            f"new_wells_avg_dist_to_neighbours_within_{cfg.neighbor_radius:g}":
                float(new[f"avg_dist_within_{cfg.neighbor_radius:g}"].mean()) if n_new else float("nan"),
            f"existing_wells_avg_dist_to_neighbours_within_{cfg.neighbor_radius:g}":
                float(w.loc[w.status == "existing", f"avg_dist_within_{cfg.neighbor_radius:g}"].mean())
                if (w.status == "existing").any() else float("nan"),
        }
        if res.quality_map is not None:
            qn = res.quality_map.name
            prod_q = new.loc[new.role == PRODUCER, qn]
            inj_q = new.loc[new.role == INJECTOR, qn]
            s[f"new_producers_min_{qn}"] = float(prod_q.min()) if len(prod_q) else float("nan")
            s[f"new_producers_mean_{qn}"] = float(prod_q.mean()) if len(prod_q) else float("nan")
            s[f"new_injectors_mean_{qn}"] = float(inj_q.mean()) if len(inj_q) else float("nan")
        if lattice_info:
            s["lattice_rotation_deg"] = lattice_info["rotation_deg"]
        if s["spacing_violations"]:
            log.warning("%d new wells violate the spacing constraint", s["spacing_violations"])
        return s


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate infill well locations and injector/producer roles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    io = p.add_argument_group("inputs / outputs")
    io.add_argument("--boundary", help="Boundary polygon (.csv/.xlsx vertices, .geojson, .wkt)")
    io.add_argument("--wells", help="Existing wells (.csv/.xlsx): name, X, Y, optional type/role")
    io.add_argument("--well-paths",
                    help="Trajectories of deviated/horizontal existing wells (.csv/.xlsx): "
                         "well name, X, Y, optional order column; points along each path")
    io.add_argument("--x-col", help="X column name (auto-detected if omitted)")
    io.add_argument("--y-col", help="Y column name (auto-detected if omitted)")
    io.add_argument("--out", default="infill_plan.csv", help="Output .csv or .xlsx")
    io.add_argument("--plot", help="Save a map to this PNG path")
    io.add_argument("--new-only", action="store_true", help="Export only the new wells")
    io.add_argument("--keep-outside-wells", action="store_true",
                    help="Keep existing wells outside the boundary as spacing constraints")

    pl = p.add_argument_group("planning")
    pl.add_argument("--spacing", type=float, default=750.0, help="Minimum well spacing (m)")
    pl.add_argument("--layout", choices=["hex", "square", "poisson"], default="hex")
    pl.add_argument("--standoff", type=float,
                    help="Min distance from boundary (m); default = half the spacing")
    pl.add_argument("--rotation", default="0", help="Grid rotation in degrees, or 'auto'")
    pl.add_argument("--no-gap-fill", action="store_true", help="Disable off-lattice gap filling")
    pl.add_argument("--max-new", type=int, help="Maximum number of new wells")
    pl.add_argument("--injector-ratio", default="1:2", help="e.g. 0.33, 33%%, or 1:2 (inj:prod)")
    pl.add_argument("--ratio-scope", choices=["new", "field"], default="new")
    pl.add_argument("--role-method", choices=["dispersed", "pattern", "scored"], default="dispersed")
    pl.add_argument("--producer-standoff", type=float, help="Min distance of producers from boundary (m)")
    pl.add_argument("--injector-standoff", type=float, help="Min distance of injectors from boundary (m)")
    pl.add_argument("--pay-weight", type=float, default=0.5,
                    help="0..1 weight of low quality vs spread when choosing injectors (scored)")
    pl.add_argument("--quality-map", help="CSV/XLSX with X, Y and a quality value (e.g. So or net pay)")
    pl.add_argument("--quality-col", help="Value column in --quality-map (default: first non-XY column)")
    pl.add_argument("--min-producer-quality", type=float,
                    help="Min quality for producers (e.g. 0.5 for So, 10 for net pay in m)")
    pl.add_argument("--pattern", choices=list(PATTERNS), help="Flood pattern (role-method=pattern)")
    pl.add_argument("--neighbor-radius", type=float,
                    help="Radius for neighbour-distance QC (default 1.5 x spacing)")
    pl.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> InfillResult:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s | %(message)s")

    if not args.boundary:
        raise SystemExit("--boundary is required")

    boundary = load_polygons(args.boundary, args.x_col, args.y_col)
    wells = load_wells(args.wells, args.x_col, args.y_col) if args.wells else None
    paths = load_well_paths(args.well_paths) if args.well_paths else None

    cfg = InfillConfig(
        well_spacing=args.spacing,
        layout=args.layout,
        boundary_standoff=args.standoff,
        grid_rotation_deg=None if str(args.rotation).lower() == "auto" else float(args.rotation),
        fill_gaps=not args.no_gap_fill,
        max_new_wells=args.max_new,
        injector_ratio=args.injector_ratio,
        ratio_scope=args.ratio_scope,
        role_method=args.role_method,
        min_producer_quality=args.min_producer_quality,
        producer_standoff=args.producer_standoff,
        injector_standoff=args.injector_standoff,
        pay_weight=args.pay_weight,
        pattern=args.pattern,
        neighbor_radius=args.neighbor_radius,
        random_seed=args.seed,
    )
    qmap = PropertyMap.from_table(args.quality_map, args.quality_col) if args.quality_map else None
    result = InfillPlanner(boundary, wells, keep_outside_wells=args.keep_outside_wells,
                           well_paths=paths).run(cfg, qmap)
    result.print_summary()
    result.to_file(args.out, new_only=args.new_only)
    if args.plot:
        result.plot(args.plot)
    return result


def cli() -> int:
    """Console-script entry point (returns an exit code, not the result object)."""
    main()
    return 0


if __name__ == "__main__":
    cli()
