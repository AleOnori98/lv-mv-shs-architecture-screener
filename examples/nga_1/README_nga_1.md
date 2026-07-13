# NGA 1 Compact Village Example

This folder contains a complete MVP-ready input set derived from the raw NGA 1 GIS assets:

- `Consumers.gpkg`: original 128 customer point locations, preserved as the georeferenced customer set.
- `Roads.gpkg`: original road layout, also exported to `roads.geojson` for the app loader.
- `Aggregated Demand.csv`: 8760-hour aggregate village demand profile.

The generated `customers.csv` keeps the original consumer coordinates and maps all raw
`household` consumers to the MVP customer schema. Because the source layer does not provide
per-customer demand, the aggregate profile is distributed evenly across the 128 households:

- total present peak: approximately 16.7 kW
- total annual demand: approximately 49.3 MWh/year
- demand tier: 2
- power factor: 0.95
- annual growth: 4%
- `must_connect_flag`: false
- `can_be_standalone_flag`: true

The source point is placed at the customer centroid snapped to the nearest road. The LV/MV
conductor catalogues, transformer catalogue, stand-alone package catalogue and cost parameters
are copied from the Mafa case study so the two Nigerian examples remain directly comparable.
