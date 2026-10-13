# Release Notes

* [3.6](#36-release)    
* [3.5](#35-release)
* [3.4](#34-release)
* [3.3.1](#331-release)
* [3.3.0](#330-release)
* [3.2.0](#320-release)
* [3.1](#31-release)
* [3.0.1](#301-release)
* [3.0](#30-release)
* [2.2](#22-release)
* [2.1](#21-release)
* [2.0](#20-release)
* [1.2](#12-release)
* [1.1](#11-release)
* [1.0](#10-release)
* [0.9](#09-release)
* [0.8](#08-release)


## 3.6 Release

### Description

This minor release adds an MCP (Model Context Protocol) server for Kosh.

### New in this release    

The Kosh MCP server is built as an extension of the mada-tools server suite. It lets AI agents and agentic workflows query, access, and update objects in Kosh stores through standard MCP tool calls with no custom glue code.
Tools:
    * Store connection: connect to a Kosh store
    * Find: search for datasets, ensembles, and other objects
    * Create: create datasets or ensembles
    * Associate/dissociate: attach file to, or detach them from, datasets
    * Update: modify datasets, ensembles, or source records
    * Load: list, describe, or load feature data from a dataset
    * Transform: apply Kosh transformers or operators
    * Import/export: import native Kosh export files, or export datasets and ensembles
    * Dataframe: preview or save Kosh dataframe output
    * Add curve: add curve data to a dataset
    * Delete: remove objects from a store    
    
## 3.5 Release

### Description

This minor release adds support for specifying column data types and replacement values when converting Kosh database data to Pandas DataFrames.

### New in this release

  * Added options to specify column data types in `.to_dataframe()`.
  * Added options to replace column values during `.to_dataframe()` conversion.

### Improvements

* None

### Bug fixes

* None

## 3.4 Release

### Description

This is a minor release which adds parameterized workflow tracking and includes clustering and store robustness fixes.

### New in this release

* Added the `kosh_workflow` command line tool for creating, querying, and updating datasets from a workflow step plus a parameter set.
* Added the `kosh.parameter_store.StepRequest` and `kosh.parameter_store.apply_step(...)` Python API for the same workflow-tracking behavior in scripts.
* Workflow matching supports optional ensemble scoping, exact numeric matching with `--strict-match`, tolerance-based numeric matching with `--rtol` and `--atol`, duplicate avoidance with `--upsert`/`--upsert-init`, dataset and ensemble metadata updates, and file associations.
* `kosh_workflow --check` reports matching dataset ids, current workflow step, and last step-update timestamp; `kosh_workflow --size` reports store or ensemble dataset counts.
* Added `examples/Example_Workflow_Manager.ipynb` and documentation references for parameter-set workflow management.
* Packaged the Kosh Codex skill files and documented workflow tracking guidance for agent use.

### Improvements

* Kosh now requires Sina 1.16.0 or greater.
* Modernized package metadata through `pyproject.toml` and entry points for `kosh` and `kosh_workflow`.
* Improved cleanup on store close to reduce SQLAlchemy/SQLite shutdown warnings when connections are finalized from a different thread.
* Made Kosh user-record creation safer under concurrent access by treating confirmed duplicate user records as success.
* Normalized empty MPI clustering index results to an empty array instead of `None`, which makes downstream gather/concatenate operations more reliable.

### Bug fixes

* Fixed clustering failures caused by single-value NumPy arrays being passed where scalar distance thresholds or loss values were required.
* Made batch clustering tests compare against a non-batched reference instead of dependency-sensitive hard-coded sample counts.

## 3.3.1 Release
 
### Description
 
This is a patch release which includes a couple of bug fixes.
 
### New in this release
 
* Kosh now requires Sina 1.16.0
 
### Bug fixes
 
* Test when reading json files and decoding them
* Read HDF5 function name updated
* Missing `"name"`, `"creator"`, `"creation_date"` in `import_dataset()`

## 3.3.0 Release

### Description

This is a minor release which includes a couple of new features and bug fixes.

### New in this release

* `store.import_dataset()` now works with Record, Pandas DataFrame, CSV, and HDF5 objects/files.
* Users can now sort (default is ascending) by an attribute in the `.find(session_sort_by=None, session_sort_by_descending=False)` methods.
* Users can now sort (default is ascending) by an attribute at the store session level `store.connect(session_sort_by=None, session_sort_by_descending=False)`.

### Improvements

* Can now export `dataset` as HDF5 file
* `"creation_date"` and `"last_modified_date"` were added to default `dataset` attributes.
* Ultra loader now defaults to PyDV curve objects for user familarity.
* `examples/Example_Simulation_Workflow.ipynb` notebook has been updated with more examples.

### Bug fixes

* Flake8 Record global import.
* Records of the type 'run' were not ingestable by Kosh.
* Garbage collection on store closure with new lock strategies.

## 3.2.0 Release

### Description

This release includes several new defining features that improve Kosh data management and functionality.

### New in this release

* Pandas read functions were added as loader types and added examples to notebooks.
* `examples/Example_Simulation_Workflow.ipynb` notebook shows the user how to add, update, and access data for Kosh in a generic simulation workflow.
* `examples/MPI_Kosh_data_management.ipynb` notebook shows the user how to load and operate on large datasets in a distributed way.
* Convert `.find()` methods to Pandas DataFrames
* L1, L2 and L* operators.
* Organize datasets within ensembles with `ensemble_tags` which also work with `ensemble.find_datasets(data=target_data, ensemble_tags=target_ensemble_tags))`.
* Lock Strategies for parallel calls see: `examples/Example_ThreadSafe.ipynb`
* Added `connection_type='append'` which allows the user to only add data but not modify or delete data.

### Improvements

* `koshClustering` operator can return a non-dimensional loss, distribute data from one rank, and added example updates.
* `dataset.list_features(use_cache=True)` allows for faster listing of features since they are now cached.
* original `import sina` is much faster as most imports are delayed until actually needed (lazy imports).
* When printing extremely long strings stored as attributes, the terminal output is overwhelming for users. We now utilize the reprlib library.

### Bug fixes

* Fixes to internal tests
* tar `-v` command is now accepted
* Some special sina files couldn't be imported by Kosh (e.g no `data` section)

## 3.1 Release

### Description

This release is a minor release with a few bug fixes and new features. We encourage users to upgrade.

### New in this release

* When searching for datasets from the store you can request to return them as Kosh Datasets (default no change in behaviour), Sina records, or Python dictionaries, which enables faster returns in loops. `store.find([...],load_type='dictionary')`.
* Kosh stores have an `alias_feature` attribute, that is used to allow users to extract features via an aliased name.
* New Auto Epsilon Algorithm for Clustering: The algorithm will find the right epsilon value to use in clustering. The user can specify the amount of allowed information loss due to removing samples from the dataset.
* Requires sina >=1.14
* When opening a mariadb backend, in order to avoid sync error between ranks you should use: `store = kosh.connect(mariadb, execution_options={"isolation_level": "READ COMMITTED"})`
* `mv` and `cp` from the command line now have `--merge_strategy` and `mk_dirs` options
* `cp`, `mv` and `tar` are now accessible from Python at the store level: `store.cp()`, `store.mv()` and `store.tar()`
* There is a [README](tests/README.md) for the Kosh test suite, including a dedicated one for [LC users](tests/LC_README.md)
* Sina new ingest capabilities are available in Kosh via dataset, but with decoartor to allow the use of functions operating on Sina records.
* Documentation switched to mkdocs.

### Improvements

* Some internal cleanups (internal kosh attributes are being moved to their own section under the `user_defined`` section of the sina record).
* Clustering now has a verbose option.
* When using MPI the clustering can be gathered to your prefered rank (rather than 0) with `gather_to`
* Batch clustering has a more lenient convergence option resulting in faster clustering sampling.
* Getting a warning when a loader cannot be loaded into the store.
* Using bash rather than sh for the sbang
* `latin1` encoding of loaders seems to create issues with mariadb, switching to `windows-1252`
* Test suite gets mariadb from env variable.
* Issue a warning if trying to set an ensemble attribute from a dataset and it matches the existing value. It still produces an error if the values differ.
* KoshCluster is more consistent in what it returns. It will always return a list now even if None is returned.

### Bug fixes

* Kosh parallel clustering used to hang when sample size was too small.
* Kosh parallel clustering returned indices as a 1D array rather than a flat array.
* On BlueOS `update_json_file_with_records_and_relationships` used to fail.
* Reassociating a file linked to many datasets used to fail for other datasets if the reassociation was done at the dataset level.
* `use_lock_file` caused hanging while using mariadb.
* `mv` command now works with nested dirs
* `mv` and `cp` now preserve ensemble membership.
* KoshClustering `operate` uses inputs shape rather than original datasets sizes.

## 3.0.1 Release

### Description

This release is a patch release.

### New in this release

* Store can open a dataset based on a Sina record (`store.open(sina_record)`).

### Improvements

* Copyright for `compute_hopkins_statistic`
* Uses sinas `exist` function rather than a `try`/`except` to decide if we update or insert new records into the store.

### Bug fixes

* None

## 3.0 Release

### Description

This release introduces clustering capabilities into Kosh. It also drops support for Python 2.

### New in this release

* Support for Clustering (via operators).
* Dropped Python 2 support.
* Loaders can access the dataset requesting the data
* Operators now have a `describe_entries` function to help them understand what's coming to them.

### Improvements

* `find` function accepts `id` as an alias for `id_pool` to restrict search to some ids
* The store can now be used within a context manager
* Curves can be added/removed to a dataset
* Ensembles can be created from the command line
* Passing `type=None` when searching the store will return Kosh specific objects as well as regular datasets (e.g associated files objects)
* Added a `verbose` mode to `dataset.list_features()` to let users know when a loader failed to load a uri. Mostly useful for debugging


### Bug fixes

* Fixed an issue where associating a file multiple time was not reflected into the store and a subsequent dissociation or an async association would not be caught. Dissociation would cause the object to be removed from the store.
* Deleting a dataset attribute and re-adding it would cause a crash
* Loaders can be removed from store

## 2.2 Release
### Description

This is a maintenace release, with added support for Windows systems.

### New in this release

* Support for Windows systems (note that `kosh cp` and `kosh mv` are not currently supported on Windows)
* While importing sina `json` files you can now skip over some sections, such as `curve_sets`.

### Improvements

* `find` function accepts `id` as an alias for `id_pool` to restrict search to some ids

### Bug fixes

* Fixed an issue where associating a file multiple time was not reflected into the store and a subsequent dissociation or an async association would not be caught. Dissociation would cause the object to be removed from the store.

## 2.1 Release

### Description

This is a mainly maintenance release introducing a few new features.

### New in this release

* Decorators for transformers and operators
  * Regular `def foo(...):` can now be converted to transformers or operators via decorators. e.g for a numpy transformer: `@kosh.numpy_transformer`. See the [transformer](examples/Example_05a_Transformers.ipynb) and [operators](examples/Example_06_Operators.ipynb) notebooks for more details.
* Introducing a loader for text and column-based data, based on top `numpy.loadtxt`. See [this](examples/Example_column_based_text_files.ipynb) notebook.
* When cloning a dataset one can choose to preserve ensemble memberships (`preserve_ensembles_memberships=True`), simply copy over these attributes (`preserve_ensembles_memberships=False`) (**default**) or ignore information related to ensembles (`preserve_ensembles_memberships=-1`)
* Dataset objects now have a `is_ensemble_attribute()` function to know if an attribute belongs to an ensemble.
* Added `list_attributes()` function to ensemble objects.

### Improvements

* When printing a dataset, the attributes coming from ensembles the dataset belongs to are listed in a separate section.

### Bug fixes

* When cloning a dataset, an artificial `id` attribute was created with the original dataset `id` in it.
* Setting an attribute to an invalid value would cause the dataset to disappear from the store.

## 2.0 Release

### Description

This release aligns Kosh with Sina and makes it the only backend. Kosh and Sina (1.11) API's have been mostly aligned.
Sina curve and file section can now be recognized and taken advantage by Kosh.

### New in this release

* Sina alignment:
  * Sina is only supported backend, no more code to potentially support other backends
  * Curves appear as associated
  * Files with `mimetype` appear as virtual associated files
  * Kosh exported json files are sina-compatible and Kosh can ingest Sina's json files
  * Stores are now opened via: `kosh.connect(...)`
  * `search(...)` is now  `find(...)`
  * Any non Kosh-reserved record type is considered a dataset
* `find` functions return generators (used to be lists)
* Support for ensembles
* Kosh stores can be associated with other Kosh stores.
* `kosh` command line:
  * Can create stores
  * Can add datasets
  * Can use htar to tar up data
* Datasets can be cloned
* While importing a dataset into a store, there are now options to handle conflicts.
* Loader for file saved by numpy (`.npy`)
* Store can fix changed/updated fast_sha

### Improvements

* Do not try to import external Python packages until needed -> some loader might appear as valid even though python packages are missing.
* Versioning is now pip compatible
* `import_dataset(...)` can import list of datasets
* Added `verbose` argument to transformers and operators -> this will let the user know when retrieving data from cache
* Set user name to "default" if can't get it from USER env var

### Bug fixes

* `matplotlib` import would crash if no DISPLAY environment variable
* hdf5 leading / fix
* No more error if a known mime type points to missing file (`list_features` will not show it)
* Dissociate files from store after moving them


## 1.2 Release

### Description

This release is fully backward compatible but introduces new concepts.
Operators are introduced allowing the composition of features from one or many sources.
Feature selection without extraction is now possible via the new execution graphs introduced in this release.
Execution graphs are the recommended way to use Kosh going forward, as reflected in the updated notebooks.


### New in this release

* Operators: Compose multiple features (and their transformers).
* Execution graph concept (select and compose features before executing).
* New Conduit's Sidre Mesh Blueprint field loader.

## Deprecation Warning

* In future versions (not 1.2) the `search` functions will return a generator (they are currently returning a list). In this version a warning is issued when you use the `search` function.

### Improvements

* Multiple speed/caching optimizations.
* A cleanup function helps you clean your store from files that no longer exist.
* Cleaned up tables in documentation.
* Transformers get a `parent` attribute, allowing you to access its caller in the `transform` function.

### Bug fixes

* In some case where a feature was available from the loader but not listed, Kosh would let the user access it. With this bug fix, the loader's `list_features` must be fixed first in order to access the feature. This bug affected groups in HDF5 files. Groups were not listed but still accessible, HDF5 loader now lists `groups` as actual features.
* If a loader needed matplotlib and no X connection was available the import of `matplotlib.pyplot` would lead to an uncatchable error. We now check for a valid backend first (via environment's `DISPLAY` variable).

## 1.1 Release

### Description

This is a maintenance release, with a few small bugs fixes and optimizations.
A new object to help you drive existing scripts from searches in Kosh has been introduced.


### New in this release

* New script wrapper object allows you to drive existing command line based scripts using Kosh objects.
* New loader: `json` files.
* Transformers `cache` option now accepts `2` as a value allowing to clobber existing cache files.
* Datasets have new `searchable_source_attributes` to list attributes that can be used to filter sources associated with this dataset.
* Can search for attribute presence (no need to specify a value).

### Improvements

* Dataset export/import optimized, `import_dataset` now accepts dataset objects.
* Warning raised if trying to associate already associated source: metadata will not be automatically updated on source.

### Bug fixes

* `list_features` cache issue cleaned up.
* Respects user passed loader.

### In the weeds

* Logos are automatically generated to reflect version.
* Uses sbang to allow executable to work in environments with very long paths.

## 1.0 Release

* Transformers are now available. These allow for post processing when extracting features.
* Transformers allow for cache
* Kosh can now move, copy or tar files tied to (a) store(s) and update store(s) appropriately.
* Documentation cleanup, logo added.
* `list_features` operation is now cached for speed improvements
* ***lock** capability improved to allow for multiple users accessing the store at same time.

## 0.9 Release

Bug fix and optimization release

* Possibility to "update" multiple attributes of a dataset at once, faster and less db access
* Possibility to associate many files at once, faster, less db access
* Creating a new store, returns a handle to that store
* Switched to pytest rather than nosetests
* Added Ultra files loader
* Loaders can be saved in store.
* When many uri were associated with a dataset and many metrics had the same name, it was not necessarily returning the desired metric even when asking for one by its full path.

## 0.8 Release

* Command line tools (kosh program)
* Significant boost in search speed 
* Python 2 support
* Example Notebooks are self contained
* Version obtained via git
* Associated uri objects accept arguments to `open` function
* User id no longer required when opening the store (defaults to UNIX username)
* Sina type for *datasets* does not need to be `dataset` any longer.
* Can open any sina db, not necessarily Kosh generated one.
