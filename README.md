# Vertical Infill Planner

Python tool that places **vertical infill wells** inside a field boundary and assigns
each one as an **injector or producer**.

- **Spacing:** new wells keep a minimum spacing from each other and from existing wells.
  Existing deviated or horizontal wells count along their whole trajectory.
- **Standoffs:** producers and injectors each keep their own distance from the boundary,
  for example the oil-water contact.
- **Reservoir-quality screening (optional):** producers only go where the map shows
  enough oil (net pay, So, ...).
- **Injector ratio:** applies to the new wells alone or to the whole field.

![Example: Norne field](docs/norne_infill_plans.png)
*Example application on the public Norne field model (see `examples/norne`).*
