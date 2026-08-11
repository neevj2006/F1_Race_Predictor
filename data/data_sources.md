# Data sources

## OpenF1

OpenF1 provided race results from 2023 through the completed races in 2026. It is also useful for session timing, laps, stints, weather and telemetry data that may be added later. Some historical result records have missing finishing positions, outdated points and missing disqualification updates.

## FastF1

FastF1, using the Jolpica results data, provided 1,640 driver results from 81 completed races: 22 races in 2023, 24 in 2024, 24 in 2025 and 11 completed races in 2026. It provides complete finishing orders, stable driver and constructor identifiers, revised points and updated disqualifications.

## Source choice

FastF1 will be used as the main source for completed race results because its historical results are more complete and better suited to normalization. OpenF1 will remain available for comparison and for detailed timing and telemetry data.

## Known correction

FastF1 lists Isack Hadjar as Retired at the 2025 Australian Grand Prix, while the official result is DNS. This is the only known exception in the collected results and will be changed to DNS during data cleaning.
