# Trino source (contrib)

## Description

Trino data sources are Trino tables or views.
These can be specified either by a table reference or a SQL query.

## Testing & Stability

The Trino data source is validated via unit and automated integration tests running against a live Trino cluster.

## Examples

Defining a Trino source:

```python
from feast.infra.offline_stores.contrib.trino_offline_store.trino_source import (
    TrinoSource,
)

driver_hourly_stats = TrinoSource(
    timestamp_field="event_timestamp",
    table="feast.driver_stats",
    created_timestamp_column="created",
)
```

The full set of configuration options is available [here](https://rtd.feast.dev/en/master/#trino-source).

## Supported Types

Trino data sources support all eight primitive types and their corresponding array types.
For a comparison against other batch data sources, please see [here](overview.md#functionality-matrix).
