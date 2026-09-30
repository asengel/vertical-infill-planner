"""
infill_planner - vertical infill well placement and injector/producer assignment.

Typical use::

    from infill_planner import InfillPlanner, InfillConfig, load_polygons, load_wells

    planner = InfillPlanner(load_polygons("boundary.csv"), load_wells("wells.csv"))
    result = planner.run(InfillConfig(well_spacing=750, injector_ratio="1:2"))
    result.to_file("plan.xlsx")
    result.plot("plan.png")
"""

from .planner import (
    PATTERNS,
    InfillConfig,
    InfillPlanner,
    InfillResult,
    PropertyMap,
    load_polygons,
    load_well_paths,
    load_wells,
    parse_ratio,
)

__all__ = [
    "PATTERNS", "InfillConfig", "InfillPlanner", "InfillResult", "PropertyMap",
    "load_polygons", "load_well_paths", "load_wells", "parse_ratio",
]
__version__ = "0.1.0"
