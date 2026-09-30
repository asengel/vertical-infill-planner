"""
infill_planner.eclipse_io
=========================

Optional readers for Eclipse-format reservoir-model files, enough to feed
the planner from any corner-point simulation deck (see examples/norne):

* corner-point grid (SPECGRID, COORD, ZCORN) and integer properties
  (ACTNUM, EQLNUM, ...) from GRDECL / .prop / .INC files
* field outline in map view: all active columns, or only the hydrocarbon
  column (cells whose top lies above the oil-water contact of their
  equilibration region)
* wells from a SCHEDULE file: WELSPECS, COMPDAT (completion cells -> map
  coordinates) and the last WCONHIST / WCONINJE control -> producer/injector

Only numpy / pandas / shapely are needed; no Eclipse licence or resdata.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

log = logging.getLogger("eclipse_io")


# --------------------------------------------------------------------------- #
# Generic keyword parsing
# --------------------------------------------------------------------------- #
def _strip_comments(text: str) -> str:
    return re.sub(r"--[^\n]*", "", text.replace("\r", ""))


def _expand(tokens: Iterable[str], dtype=float) -> np.ndarray:
    """Expand Eclipse repeat notation ('3*0.25') into a flat array."""
    out: list = []
    for tok in tokens:
        if "*" in tok:
            n, v = tok.split("*", 1)
            out.extend([dtype(v)] * int(n))
        else:
            out.append(dtype(tok))
    return np.asarray(out, dtype=dtype)


def read_grdecl_keywords(path: str | Path, names: Iterable[str]) -> dict[str, np.ndarray]:
    """Read numeric keywords (e.g. COORD, ZCORN, ACTNUM) from a GRDECL-style file."""
    text = _strip_comments(Path(path).read_text())
    out: dict[str, np.ndarray] = {}
    for name in names:
        m = re.search(rf"^\s*{name}\s*$(.*?)/", text, re.S | re.M)
        if m is None:
            continue
        toks = m.group(1).split()
        is_int = name.upper() in ("ACTNUM", "EQLNUM", "FIPNUM", "SATNUM", "PVTNUM", "SPECGRID")
        if name.upper() == "SPECGRID":
            out[name] = np.array([int(t) for t in toks[:3]])
        else:
            out[name] = _expand(toks, int if is_int else float)
    return out


# --------------------------------------------------------------------------- #
# Corner-point grid
# --------------------------------------------------------------------------- #
@dataclass
class CornerPointGrid:
    nx: int
    ny: int
    nz: int
    coord: np.ndarray          # (ny+1, nx+1, 6): x1 y1 z1 x2 y2 z2 per pillar
    zcorn: np.ndarray          # (2nz, 2ny, 2nx)
    actnum: np.ndarray         # (nz, ny, nx) bool

    @classmethod
    def from_files(cls, grid_file: str | Path, actnum_file: str | Path | None = None
                   ) -> "CornerPointGrid":
        kw = read_grdecl_keywords(grid_file, ["SPECGRID", "COORD", "ZCORN", "ACTNUM"])
        nx, ny, nz = (int(v) for v in kw["SPECGRID"])
        coord = kw["COORD"].reshape(ny + 1, nx + 1, 6)
        zcorn = kw["ZCORN"].reshape(2 * nz, 2 * ny, 2 * nx)
        act = kw.get("ACTNUM")
        if act is None and actnum_file is not None:
            act = read_grdecl_keywords(actnum_file, ["ACTNUM"])["ACTNUM"]
        if act is None:
            log.warning("No ACTNUM given: treating all cells with thickness > 0 as active")
            top, bot = zcorn[0::2], zcorn[1::2]
            act = ((bot - top).reshape(nz, ny, 2, nx, 2).mean(axis=(2, 4)) > 0).ravel()
        return cls(nx, ny, nz, coord, zcorn, np.asarray(act).reshape(nz, ny, nx).astype(bool))

    def read_property(self, path: str | Path, name: str) -> np.ndarray:
        arr = read_grdecl_keywords(path, [name])[name]
        return arr.reshape(self.nz, self.ny, self.nx)

    # --- geometry -----------------------------------------------------------
    def _pillar_xy(self, depth: np.ndarray, pi: np.ndarray, pj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """XY of pillar (pi, pj) at the given depth (linear along the pillar)."""
        c = self.coord[pj, pi]
        x1, y1, z1, x2, y2, z2 = (c[..., k] for k in range(6))
        dz = np.where(np.abs(z2 - z1) < 1e-9, 1.0, z2 - z1)
        t = (depth - z1) / dz
        return x1 + t * (x2 - x1), y1 + t * (y2 - y1)

    def cell_top_depth(self) -> np.ndarray:
        """Mean depth of each cell's top face, (nz, ny, nx)."""
        return self.zcorn[0::2].reshape(self.nz, self.ny, 2, self.nx, 2).mean(axis=(2, 4))

    def cell_centers(self) -> np.ndarray:
        """Cell centre XYZ, (nz, ny, nx, 3), from the 8 corner points."""
        z = self.zcorn.reshape(self.nz, 2, self.ny, 2, self.nx, 2)  # k,top/bot,j,dj,i,di
        xs = np.zeros((self.nz, self.ny, self.nx))
        ys = np.zeros_like(xs)
        J, I = np.meshgrid(np.arange(self.ny), np.arange(self.nx), indexing="ij")
        for dj in (0, 1):
            for di in (0, 1):
                for tb in (0, 1):
                    depth = z[:, tb, :, dj, :, di]
                    px, py = self._pillar_xy(depth, I[None] + di, J[None] + dj)
                    xs += px
                    ys += py
        return np.stack([xs / 8, ys / 8, z.mean(axis=(1, 3, 5))], axis=-1)

    def column_polygons(self, mask2d: np.ndarray, depth: float | None = None) -> list[Polygon]:
        """Map-view quadrilaterals of the (j, i) columns selected by ``mask2d``."""
        if depth is None:
            depth = float(np.median(self.cell_top_depth()[self.actnum]))
        polys = []
        for j, i in zip(*np.nonzero(mask2d)):
            pi = np.array([i, i + 1, i + 1, i])
            pj = np.array([j, j, j + 1, j + 1])
            x, y = self._pillar_xy(np.full(4, depth), pi, pj)
            polys.append(Polygon(np.column_stack([x, y])))
        return polys

    def outline(self, mask2d: np.ndarray, min_part_area: float = 0.0,
                fill_holes_below: float = 0.0, simplify: float = 0.0):
        """Union of selected columns -> (Multi)Polygon field outline."""
        geom = unary_union(self.column_polygons(mask2d)).buffer(0)
        parts = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
        cleaned = []
        for p in parts:
            if p.area < min_part_area:
                continue
            holes = [h for h in p.interiors if Polygon(h).area >= fill_holes_below]
            p = Polygon(p.exterior, holes)
            cleaned.append(p.simplify(simplify) if simplify > 0 else p)
        if not cleaned:
            raise ValueError("Outline is empty after filtering")
        return cleaned[0] if len(cleaned) == 1 else MultiPolygon(cleaned)

    # --- masks ---------------------------------------------------------------
    def active_columns(self) -> np.ndarray:
        return self.actnum.any(axis=0)

    def hydrocarbon_columns(self, eqlnum: np.ndarray, owc: dict[int, float]) -> np.ndarray:
        """Columns with at least one active cell whose top is above its region's OWC."""
        top = self.cell_top_depth()
        owc_cell = np.vectorize(lambda r: owc.get(int(r), -np.inf))(eqlnum)
        return (self.actnum & (top < owc_cell)).any(axis=0)


# --------------------------------------------------------------------------- #
# Schedule wells
# --------------------------------------------------------------------------- #
def _records(block: str) -> list[list[str]]:
    """Split a keyword block into records (lists of tokens), expanding n* defaults."""
    recs = []
    for line in block.split("/"):
        toks = re.findall(r"'[^']*'|\S+", line)
        if not toks:
            continue
        out: list[str] = []
        for t in toks:
            if re.fullmatch(r"\d+\*", t):
                out.extend(["1*"] * int(t[:-1]))
            else:
                out.append(t.strip("'"))
        recs.append(out)
    return recs


def _iter_keywords(text: str, names: set[str]):
    """Yield (keyword, records) in file order for the requested keywords."""
    lines = _strip_comments(text).split("\n")
    i = 0
    while i < len(lines):
        word = lines[i].strip().split(" ")[0] if lines[i].strip() else ""
        if word in names:
            body = []
            i += 1
            while i < len(lines) and not re.fullmatch(r"\s*/\s*", lines[i]):
                body.append(lines[i])
                i += 1
            yield word, _records("\n".join(body))
        i += 1


def read_schedule_wells(schedule_file: str | Path, grid: CornerPointGrid
                        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wells and completion cells from an Eclipse SCHEDULE file.

    Returns
    -------
    wells : one row per well - well_name, x, y (map position = completion cell
            nearest the completion centroid), role (from the *last* WCONHIST /
            WCONINJE record), fluid, n_connections, head_i, head_j
    paths : one row per completion column - well_name, x, y, i, j, order
            (sorted along the well's main direction, for plotting / spacing)
    """
    text = Path(schedule_file).read_text(errors="replace")
    centers = grid.cell_centers()

    heads: dict[str, tuple[int, int]] = {}
    conns: dict[str, list[tuple[int, int, int]]] = {}
    role: dict[str, tuple[str, str]] = {}
    for kw, recs in _iter_keywords(text, {"WELSPECS", "COMPDAT", "WCONHIST", "WCONINJE", "WCONPROD"}):
        for r in recs:
            name = r[0]
            if kw == "WELSPECS":
                heads[name] = (int(r[2]), int(r[3]))
            elif kw == "COMPDAT":
                i = int(r[1]) if r[1] not in ("0", "1*") else heads.get(name, (0, 0))[0]
                j = int(r[2]) if r[2] not in ("0", "1*") else heads.get(name, (0, 0))[1]
                k1, k2 = int(r[3]), int(r[4])
                for k in range(k1, k2 + 1):
                    conns.setdefault(name, []).append((i, j, k))
            elif kw in ("WCONHIST", "WCONPROD"):
                role[name] = ("producer", "oil")
            elif kw == "WCONINJE":
                role[name] = ("injector", r[1].lower())

    well_rows, path_rows = [], []
    for name, cl in conns.items():
        cells = list(dict.fromkeys(cl))  # unique, first-seen order
        ijk = np.array(cells) - 1        # 0-based
        ok = (ijk[:, 0] < grid.nx) & (ijk[:, 1] < grid.ny) & (ijk[:, 2] < grid.nz)
        ijk = ijk[ok]
        xyz = centers[ijk[:, 2], ijk[:, 1], ijk[:, 0]]

        # one map point per column, ordered along the principal direction
        cols = pd.DataFrame({"i": ijk[:, 0] + 1, "j": ijk[:, 1] + 1, "x": xyz[:, 0], "y": xyz[:, 1]})
        cols = cols.groupby(["i", "j"], as_index=False)[["x", "y"]].mean()
        pts = cols[["x", "y"]].to_numpy()
        if len(pts) > 1:
            d = pts - pts.mean(axis=0)
            axis = np.linalg.svd(d, full_matrices=False)[2][0]
            cols = cols.iloc[np.argsort(d @ axis)].reset_index(drop=True)
            pts = cols[["x", "y"]].to_numpy()
        rep = pts[np.argmin(np.hypot(*(pts - pts.mean(axis=0)).T))]

        r, fluid = role.get(name, ("unknown", ""))
        hi, hj = heads.get(name, (np.nan, np.nan))
        well_rows.append({"well_name": name, "x": rep[0], "y": rep[1], "role": r, "fluid": fluid,
                          "n_connections": len(ijk), "head_i": hi, "head_j": hj})
        for n, row in cols.iterrows():
            path_rows.append({"well_name": name, "order": n, "i": row.i, "j": row.j,
                              "x": row.x, "y": row.y})

    missing = sorted(set(heads) - set(conns))
    if missing:
        log.warning("Wells without COMPDAT (skipped): %s", ", ".join(missing))
    return pd.DataFrame(well_rows), pd.DataFrame(path_rows)


def read_equil(path: str | Path) -> pd.DataFrame:
    """EQUIL records -> DataFrame[region, datum, owc, goc] (region is 1-based).

    Each record is one line ``datum pressure OWC pcow GOC ... /``; text after
    the slash is ignored. The keyword ends at the next keyword or a lone '/'.
    """
    lines = _strip_comments(Path(path).read_text()).split("\n")
    try:
        start = next(i for i, l in enumerate(lines) if l.strip() == "EQUIL")
    except StopIteration:
        raise ValueError(f"No EQUIL keyword in {path}") from None
    rows = []
    for line in lines[start + 1:]:
        body = line.split("/")[0].strip()
        if re.fullmatch(r"[A-Z][A-Z0-9_]{2,7}", body) or (not body and line.strip() == "/"):
            break
        t = body.split()
        if len(t) >= 3:
            goc = float(t[4]) if len(t) >= 5 and t[4] != "1*" else float("-inf")
            rows.append({"region": len(rows) + 1, "datum": float(t[0]),
                         "owc": float(t[2]), "goc": goc})
    return pd.DataFrame(rows)


def read_equil_owc(path: str | Path) -> dict[int, float]:
    """Oil-water contact per equilibration region."""
    df = read_equil(path)
    return dict(zip(df.region.astype(int), df.owc))


# --------------------------------------------------------------------------- #
# Saturation / pay maps
# --------------------------------------------------------------------------- #
def _per_region(eqlnum: np.ndarray, table: dict[int, float], default: float) -> np.ndarray:
    lut = np.full(int(eqlnum.max()) + 1, default)
    for r, v in table.items():
        if 0 <= r < len(lut):
            lut[r] = v
    return lut[eqlnum.astype(int)]


def initial_saturations(grid: CornerPointGrid, swatinit: np.ndarray, eqlnum: np.ndarray,
                        equil: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Initial (So, Sg) per cell from SWATINIT and the EQUIL contacts.

    Above the GOC the hydrocarbon is gas (So = 0, Sg = 1 - Sw); between GOC and
    OWC it is oil (So = 1 - Sw). Below the OWC SWATINIT is ~1, so So ~ 0.
    """
    top = grid.cell_top_depth()
    bot = grid.zcorn[1::2].reshape(grid.nz, grid.ny, 2, grid.nx, 2).mean(axis=(2, 4))
    mid = 0.5 * (top + bot)
    goc = _per_region(eqlnum, dict(zip(equil.region.astype(int), equil.goc)), -np.inf)
    hc = np.clip(1.0 - swatinit, 0.0, 1.0) * grid.actnum
    in_gas = mid < goc
    return np.where(in_gas, 0.0, hc), np.where(in_gas, hc, 0.0)


def net_pay_maps(grid: CornerPointGrid, ntg: np.ndarray, so: np.ndarray,
                 sg: np.ndarray | None = None, cutoff: float = 0.5) -> dict[str, np.ndarray]:
    """Column maps (ny, nx): net oil pay and net gas pay in metres.

    Net pay = sum of NTG x cell thickness over active cells whose saturation
    is at least ``cutoff`` - i.e. the thickness a vertical well would open to
    oil (or gas).
    """
    top = grid.cell_top_depth()
    bot = grid.zcorn[1::2].reshape(grid.nz, grid.ny, 2, grid.nx, 2).mean(axis=(2, 4))
    net_h = np.clip(bot - top, 0, None) * ntg * grid.actnum
    maps = {"net_oil_pay": (net_h * (so >= cutoff)).sum(axis=0)}
    if sg is not None:
        maps["net_gas_pay"] = (net_h * (sg >= cutoff)).sum(axis=0)
    return maps


def column_property_map(grid: CornerPointGrid, values2d: np.ndarray, mask2d: np.ndarray,
                        name: str, injector_ok2d: np.ndarray | None = None):
    """Wrap a column map as an ``infill_planner.PropertyMap`` (with cell polygons)."""
    from .planner import PropertyMap

    polys = grid.column_polygons(mask2d)
    centers = np.array([p.centroid.coords[0] for p in polys])
    cell = float(np.sqrt(np.median([p.area for p in polys])))
    jj, ii = np.nonzero(mask2d)
    ok = None if injector_ok2d is None else injector_ok2d[jj, ii]
    return PropertyMap(centers, values2d[jj, ii], name=name, injector_ok=ok,
                       max_distance=1.5 * cell,
                       polygons=[np.asarray(p.exterior.coords) for p in polys])


def pore_volume_average(grid: CornerPointGrid, prop: np.ndarray, poro: np.ndarray,
                        ntg: np.ndarray | None = None) -> np.ndarray:
    """Column map (ny, nx) of ``prop`` averaged over active cells, weighted by
    net pore volume (poro x NTG x thickness). Columns without pore volume -> NaN."""
    top = grid.cell_top_depth()
    bot = grid.zcorn[1::2].reshape(grid.nz, grid.ny, 2, grid.nx, 2).mean(axis=(2, 4))
    w = np.clip(bot - top, 0, None) * poro * (1.0 if ntg is None else ntg) * grid.actnum
    tot = w.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot > 0, (w * prop).sum(axis=0) / tot, np.nan)


def top_structure(grid: CornerPointGrid) -> np.ndarray:
    """Column map (ny, nx): depth of the top of the shallowest active cell (m TVD).
    Inactive columns -> NaN."""
    top = np.where(grid.actnum, grid.cell_top_depth(), np.inf).min(axis=0)
    return np.where(np.isfinite(top), top, np.nan)
