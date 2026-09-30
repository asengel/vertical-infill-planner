# Example: Norne field

Applies the vertical infill planner to the public Norne model (Equinor/OPM).

```bash
python examples/norne/run_norne.py
```

On first run, the grid, schedule, ACTNUM, EQLNUM, EQUIL, NTG and SWATINIT files are
downloaded from the [OPM opm-data repository](https://github.com/OPM/opm-data/tree/master/norne)
into `examples/norne/data/`. Results go to `examples/norne/output/`.

Workflow:
1. **Boundary:** outline of the hydrocarbon columns (cells above the oil-water contact of their EQUIL region).
2. **Quality map:** net oil pay, from SWATINIT and the contacts (So ≥ 0.5).
3. **Existing wells:** from COMPDAT, with full trajectories; roles from the last WCONHIST/WCONINJE record.
4. **Plans:** one per spacing in `SPACINGS`, with scored roles and a field-level injector ratio.
5. **Figure:** one map per spacing on the top-reservoir depth.

Settings (spacing, ratio, standoffs, pay cutoff) are at the top of `run_norne.py`.

Data licence: Norne © Equinor, Open Database License (ODbL).
