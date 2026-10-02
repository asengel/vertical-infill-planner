# Example: Norne field

Applies the vertical infill planner to the public Norne model (Equinor/OPM).

```bash
python examples/norne/run_norne.py
```

On first run, the grid, schedule, ACTNUM, EQLNUM, EQUIL, NTG and SWATINIT files are
downloaded from the [OPM opm-data repository](https://github.com/OPM/opm-data/tree/master/norne)
into `examples/norne/data/`. Results go to `examples/norne/output/`.

Settings (spacing, ratio, standoffs, pay cutoff) are at the top of `run_norne.py`.

Data licence: Norne © Equinor, Open Database License (ODbL).
