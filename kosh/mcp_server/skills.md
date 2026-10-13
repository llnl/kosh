# Kosh MCP Usage Patterns

This guide covers common Kosh MCP workflows using the current tool surface. Most
tools require explicit object context through `object_type` or `owner_type` so
dataset and ensemble operations stay unambiguous.

## Switching Stores

Use `store_connection` when the user wants the active Kosh MCP server to point
at a different store, or when they ask which store is currently active:

```
store_connection(action="current")
store_connection(action="connect", alias="archive")
store_connection(action="connect", database_uri="/tmp/experiment.sqlite")
```

Use `action="current"` to inspect the current connection. Use
`action="connect"` with exactly one of `alias` or `database_uri` to switch
stores. Aliases come from the server-managed `KOSH_STORE_ALIASES_JSON`
setting. The server keeps the current `KOSH_CONNECTION_TYPE` and
`KOSH_EXECUTION_OPTIONS` when switching stores and returns an absolute local
`full_path` when the target is file-backed.

## Working with Curves

Use `add_curve` to add paired 1D arrays directly to datasets without associating
files:

```
1. create_dataset_or_ensemble(object_type="dataset", name="sweep_results")
2. add_curve(dataset_id="sweep_results", name="temperature", x_values=[...], y_values=[...])
3. load(dataset_id="sweep_results", feature_name="temperature")
```

Curves are useful for computed results, synthetic data, time series, and
parameter sweeps that do not need a separate file.

## Ensemble Management

Create an ensemble, create datasets, then update dataset membership explicitly:

```
1. create_dataset_or_ensemble(object_type="ensemble", name="simulation_run")
2. create_dataset_or_ensemble(object_type="dataset", name="sim_001")
3. update(object_type="dataset", object_id="sim_001", add_to_ensemble="simulation_run")
4. find(object_type="dataset", ensemble_id="simulation_run")
```

Set ensemble tags when adding a dataset to an ensemble:

```
update(
    object_type="dataset",
    object_id="sim_001",
    add_to_ensemble="simulation_run",
    ensemble_tags={"phase": "validation", "split": "holdout"}
)
```

Use `inherit_attributes=False` when a dataset may keep attributes that conflict
with ensemble attributes:

```
update(
    object_type="dataset",
    object_id="sim_001",
    add_to_ensemble="simulation_run",
    inherit_attributes=False
)
```

Set ensemble metadata with `update(object_type="ensemble", ...)`:

```
update(
    object_type="ensemble",
    object_id="simulation_run",
    attributes={"campaign": "baseline"}
)
```

## File Association

Associate files with either a dataset or an ensemble by providing `owner_type`:

```
associate_or_dissociate_file(
    owner_type="dataset",
    owner_id="sim_001",
    file_uris=["path/to/file1.h5", "path/to/file2.h5"],
    mime_types=["hdf5", "hdf5"],
    metadata=[{"run": 1}, {"run": 2}]
)
```

Use one MIME type for every file or provide one MIME type per file. Metadata may
be a single object applied to the association or one object per file.

Use Kosh association options when a loader needs extra configuration or when
robust file reassociation is important:

```
associate_or_dissociate_file(
    owner_type="dataset",
    owner_id="sim_001",
    file_uris=["table.csv"],
    mime_types="pandas/csv",
    loader_kwargs={"index_col": 0},
    long_sha=True,
    preload_features=True
)
```

`loader_kwargs` applies to every file in the call and cannot be combined with a
per-file metadata list. Association responses include the Kosh source IDs.

Update source metadata later by using the source ID returned from association or
from `find(object_type="source", ...)`:

```
update(
    object_type="source",
    object_id="source_id",
    attributes={"calibration": "v2", "loader_kwargs": {"index_col": None}}
)
```

Source `uri`, `mime_type`, and `associated` fields are identity/relationship
fields and cannot be updated with `update`.

Remove associations without deleting files on disk:

```
associate_or_dissociate_file(
    owner_type="dataset",
    owner_id="sim_001",
    file_uris=["path/to/file1.h5"],
    operation="dissociate"
)
```

## Query Patterns

Find datasets, ensembles, or source/file records with explicit `object_type`:

```
find(object_type="dataset", ids=["sim_001", "sim_002"])
find(object_type="ensemble", data_criteria={"campaign": "baseline"})
find(object_type="source", file_uri="path/to/file1.h5")
```

Use range criteria for numeric searches:

```
find(
    object_type="dataset",
    data_criteria={
        "temperature": {"min": 273, "max": 373},
        "pressure": {"min": 100000}
    },
    mode="objects",
    page_size=25
)
```

Scope source queries to a specific owner with both `owner_type` and `owner_id`:

```
find(
    object_type="source",
    owner_type="dataset",
    owner_id="sim_001"
)
```

Combine filters when the Kosh API supports them:

```
find(
    object_type="dataset",
    ensemble_id="simulation_run",
    data_criteria={"converged": True, "error": {"max": 0.01}},
    mode="objects",
    page_size=10
)
```

For broad discovery questions, prefer `count` or `summary` instead of asking for
many objects:

```
find(object_type="dataset", mode="count")
find(object_type="ensemble", mode="summary", page_size=20)
find(object_type="dataset", ensemble_id="simulation_run", mode="summary", page_size=20)
```

Avoid `allow_expensive=True` unless the user explicitly wants a potentially
expensive scan on the current store.

## Loading Feature Data

List available loadable features before requesting data:

```
load(dataset_id="sim_001", action="list_features")
```

List features from a specific associated source when duplicate feature names or
loader-specific options matter:

```
load(
    dataset_id="sim_001",
    action="list_features",
    source_id="source_id",
    options={"group": "node"}
)
```

Describe a feature before loading it:

```
load(dataset_id="sim_001", action="describe_feature", feature_name="density")
```

Load one feature, multiple features, or a grouped feature request:

```
load(dataset_id="sim_001", feature_name="density", output_format="numpy")
load(dataset_id="sim_001", feature_name=["time", "density"], group=True)
```

`list_features=True` still works as a backward-compatible alias for
`action="list_features"`. Feature payloads are bounded and JSON-safe. Use
`dataframe` when you need tabular metadata output instead of feature data.

## Transforming Feature Data

List supported built-in Kosh transformers and operators:

```
transform(action="list")
```

Apply transformers to a feature graph:

```
transform(
    inputs={
        "dataset_id": "sim_001",
        "feature_name": "density",
        "transformers": [
            {"name": "StandardScaler"},
            {"name": "Delta", "options": {"axis": 0}}
        ]
    },
    output_format="numpy"
)
```

Use `KoshLNorm` to combine multiple feature graphs:

```
transform(
    inputs=[
        {"dataset_id": "sim_001", "feature_name": "x0"},
        {"dataset_id": "sim_001", "feature_name": "y0"},
        {"dataset_id": "sim_002", "feature_name": "x1"},
        {"dataset_id": "sim_002", "feature_name": "y1"}
    ],
    operator={"name": "KoshLNorm", "options": {"power": 2, "overlap_only": True}},
    output_format="numpy"
)
```

Transformer input specs also accept `source_id`, `use_cache`, and
loader-specific `options`. Use multiple input specs for multiple feature
graphs. Top-level `options` are passed to graph traversal.

## Native Import and Export

Use `import_or_export` for Kosh-native importable files.

Export one object to a file:

```
import_or_export(
    action="export",
    path="exports/sim_001.json",
    object_type="dataset",
    object_ids="sim_001"
)
```

Export multiple objects to a directory. The server writes one native Kosh file
per object:

```
import_or_export(
    action="export",
    path="exports/",
    object_type="dataset",
    object_ids=["sim_001", "sim_002"]
)
```

Import a native Kosh export file:

```
import_or_export(
    action="import",
    path="exports/sim_001.json",
    match_attributes=["name"],
    merge_handler="preserve"
)
```

`object_type` is required for exports and ignored for imports.

## DataFrame Output

Use `dataframe` for tabular previews or saved CSV/JSON output. It supports
`scope="store"`, `scope="dataset"`, and `scope="ensemble"`.

Preview store metadata:

```
dataframe(action="preview", scope="store", data_columns=["status", "shot"])
```

Save a dataset dataframe:

```
dataframe(
    action="save",
    scope="dataset",
    object_id="sim_001",
    output_path="exports/sim_001_dataframe.csv",
    output_format="csv",
    overwrite=True
)
```

Preview an ensemble dataframe with ensemble filters:

```
dataframe(
    action="preview",
    scope="ensemble",
    object_id="simulation_run",
    data_criteria={"converged": True},
    ensemble_tags={"phase": "validation"}
)
```

## Update and Cleanup

Always provide `object_type` for updates and deletes:

```
update(
    object_type="dataset",
    object_id="sim_001",
    attributes={"status": "processed", "quality": 0.95}
)

delete(object_type="dataset", object_id="sim_001")
```

Deleting a dataset or ensemble removes the database entry, not associated files
on disk. Remove ensemble membership before deletion when you need a specific
cleanup order:

```
1. update(object_type="dataset", object_id="sim_001", remove_from_ensemble="simulation_run")
2. delete(object_type="dataset", object_id="sim_001")
```

## Common Workflows

New simulation result:

```
1. create_dataset_or_ensemble(object_type="dataset", name="sim_001")
2. associate_or_dissociate_file(owner_type="dataset", owner_id="sim_001", file_uris=["output.h5"], mime_types="hdf5")
3. update(object_type="dataset", object_id="sim_001", attributes={"version": "1.0"})
4. update(object_type="dataset", object_id="sim_001", add_to_ensemble="current_run")
```

Post-processing:

```
1. find(object_type="dataset", ensemble_id="raw_data", mode="ids", page_size=100)
2. load(dataset_id="sim_001", feature_name="raw_signal")
3. add_curve(dataset_id="sim_001", name="filtered_signal", x_values=[...], y_values=[...])
4. dataframe(action="save", scope="dataset", object_id="sim_001", output_path="exports/sim_001.csv")
```

Data discovery:

```
1. find(object_type="ensemble", mode="summary", page_size=20)
2. find(object_type="dataset", ensemble_id="selected_ensemble", mode="summary", page_size=20)
3. find(object_type="source", owner_type="dataset", owner_id="sim_001")
4. load(dataset_id="sim_001", list_features=True)
```
