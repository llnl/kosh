"""Kosh MCP server for scientific data management."""

import json
import logging
import sys
from typing import Any, Dict, List, Optional, Union

from .operations import KoshMCPOperations
from .utils import (
    environment_value,
    execution_options_from_environment,
    get_env_var,
    normalize_store_aliases,
    set_loglevel,
)


MCP_EXTRA_INSTALL_MESSAGE = (
    "Kosh MCP support requires the optional 'mcp' extra. "
    "Install it with: pip install 'kosh[mcp]'"
)


class KoshMCPDependencyError(RuntimeError):
    """Raised when optional MCP server dependencies are not installed."""


def _load_base_server():
    """Import mada-tools only when MCP server functionality is requested."""
    try:
        from mada_tools.shared.base_server import BaseMCPServer
    except ImportError as e:
        raise KoshMCPDependencyError(MCP_EXTRA_INSTALL_MESSAGE) from e
    return BaseMCPServer


class _KoshServerMixin:
    """Expose the Kosh MCP tool surface.

    The mixin registers these tools:

    * ``store_connection`` for switching or inspecting the active store
    * ``find`` for querying datasets, ensembles, and sources
    * ``create_dataset_or_ensemble`` for object creation
    * ``associate_or_dissociate_file`` for file association management
    * ``update`` for metadata and ensemble membership changes
    * ``load`` for feature discovery and retrieval
    * ``transform`` for execution-graph transforms and operators
    * ``import_or_export`` for native Kosh import and export
    * ``dataframe`` for dataframe preview and persistence
    * ``add_curve`` for direct 1D curve ingestion
    * ``delete`` for object deletion
    """

    DEFAULT_DATABASE_URI = KoshMCPOperations.DEFAULT_DATABASE_URI
    DEFAULT_LOGLEVEL = "WARNING"
    DEFAULT_CONNECTION_TYPE = KoshMCPOperations.DEFAULT_CONNECTION_TYPE
    ENV_DATABASE_URI = "KOSH_DATABASE_URI"
    ENV_CONNECTION_TYPE = "KOSH_CONNECTION_TYPE"
    ENV_EXECUTION_OPTIONS = "KOSH_EXECUTION_OPTIONS"
    ENV_STORE_ALIASES = "KOSH_STORE_ALIASES_JSON"
    ENV_LOGLEVEL = "KOSH_LOGLEVEL"

    def __init__(self):
        """Initialize the Kosh server."""
        super().__init__(
            server_name="Kosh Data Manager",
            description="Simplified Kosh data storage and querying",
        )

        log_level = getattr(logging, self.DEFAULT_LOGLEVEL)
        logging.basicConfig(level=log_level)
        self.logger = logging.getLogger("kosh_mcp_server")

        self.database_uri = self.DEFAULT_DATABASE_URI
        self.connection_type = self.DEFAULT_CONNECTION_TYPE
        self.execution_options: Dict[str, Any] = {}
        self.store_aliases: Dict[str, str] = {}
        self.operations = KoshMCPOperations(
            database_uri=self.database_uri,
            connection_type=self.connection_type,
            execution_options=self.execution_options,
            store_aliases=self.store_aliases,
            logger=self.logger,
        )

    def _configure_operations_from_environment(self) -> None:
        """Apply Kosh runtime settings from environment variables."""
        self.database_uri = environment_value(
            self.ENV_DATABASE_URI,
            fallback=self.DEFAULT_DATABASE_URI,
        )
        self.connection_type = environment_value(
            self.ENV_CONNECTION_TYPE,
            fallback=self.DEFAULT_CONNECTION_TYPE,
        )
        self.execution_options = execution_options_from_environment(
            self.ENV_EXECUTION_OPTIONS
        )
        store_aliases_value = get_env_var(self.ENV_STORE_ALIASES)
        if store_aliases_value is None or not store_aliases_value.strip():
            self.store_aliases = {}
        else:
            try:
                store_aliases = json.loads(store_aliases_value)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{self.ENV_STORE_ALIASES} must be a JSON object mapping aliases to database URIs."
                ) from e
            self.store_aliases = normalize_store_aliases(
                store_aliases,
                param_name=self.ENV_STORE_ALIASES,
            )
        self.operations.configure(
            database_uri=self.database_uri,
            connection_type=self.connection_type,
            execution_options=self.execution_options,
            store_aliases=self.store_aliases,
        )
        set_loglevel(
            self.logger,
            environment_value(
                self.ENV_LOGLEVEL,
                fallback=self.DEFAULT_LOGLEVEL,
            )
        )

    def _register_tools(self):
        """Register all MCP tools."""
        self._configure_operations_from_environment()
        self.operations.validate_connection()

        @self.mcp.tool()
        async def store_connection(
            action: str = "current",
            alias: Optional[str] = None,
            database_uri: Optional[str] = None,
        ) -> Any:
            """Inspect or switch the active Kosh store used by later tool calls.

            Use ``action="current"`` to inspect the active store and
            ``action="connect"`` with exactly one of ``alias`` or
            ``database_uri`` to switch stores. Aliases come from
            ``KOSH_STORE_ALIASES_JSON``. The server keeps the configured
            ``connection_type`` and ``execution_options`` unchanged.

            :param action: ``"current"`` or ``"connect"``.
            :param alias: Preconfigured store alias to resolve server-side.
            :param database_uri: Raw Kosh database URI or local path.
            :returns: A JSON-safe mapping describing the active store,
                including an absolute local ``full_path`` when available.
            """
            payload = await self.run_tool(
                self.operations.store_connection,
                action=action,
                alias=alias,
                database_uri=database_uri,
                background=False,
            )
            self.database_uri = self.operations.database_uri
            return payload

        @self.mcp.tool()
        async def find(
            object_type: str,
            ids: Optional[Union[str, List[str]]] = None,
            data_criteria: Optional[Dict[str, Any]] = None,
            ensemble_id: Optional[str] = None,
            file_uri: Optional[str] = None,
            owner_type: Optional[str] = None,
            owner_id: Optional[str] = None,
            mode: str = "auto",
            page_size: int = KoshMCPOperations.DEFAULT_FIND_PAGE_SIZE,
            detail: str = "compact",
            allow_expensive: bool = False,
        ) -> Any:
            """Query datasets, ensembles, or source records.

            Supported filters include ID lookup, data criteria, ensemble
            membership, associated file URI lookup, and source ownership.
            Broad discovery queries default to compact summary mode so the
            model can ask "how many?" without dumping many full objects.

            :param object_type: ``"dataset"``, ``"ensemble"``, or ``"source"``.
            :param ids: Single ID or list of IDs to retrieve.
            :param data_criteria: Field criteria, including range queries such
                as ``{"field": {"min": 0, "max": 100}}``.
            :param ensemble_id: Ensemble reference used to filter datasets by
                membership.
            :param file_uri: Associated file URI filter for datasets or sources.
            :param owner_type: ``"dataset"`` or ``"ensemble"`` when scoping
                source queries to an owner.
            :param owner_id: Dataset or ensemble ID or name for owner-scoped
                source queries.
            :param mode: ``"auto"``, ``"summary"``, ``"count"``, ``"ids"``,
                or ``"objects"``.
            :param page_size: Maximum number of results to return for
                non-count modes.
            :param detail: ``"compact"`` or ``"standard"`` object summaries.
            :param allow_expensive: Opt in to full-store scans when no cheap
                count/page path is available.
            :returns: A JSON-safe mapping containing the query results.
            """
            return await self.run_tool(
                self.operations.find,
                object_type=object_type,
                ids=ids,
                data_criteria=data_criteria,
                ensemble_id=ensemble_id,
                file_uri=file_uri,
                owner_type=owner_type,
                owner_id=owner_id,
                mode=mode,
                page_size=page_size,
                detail=detail,
                allow_expensive=allow_expensive,
                background=False,
            )

        @self.mcp.tool()
        async def create_dataset_or_ensemble(
            object_type: str,
            name: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None,
        ) -> Any:
            """Create a new dataset or ensemble.

            :param object_type: ``"dataset"`` or ``"ensemble"``.
            :param name: Optional name for the new object.
            :param metadata: Additional metadata attributes.
            :returns: A JSON-safe mapping describing the created object.
            """
            return await self.run_tool(
                self.operations.create_dataset_or_ensemble,
                object_type,
                name=name,
                metadata=metadata,
                background=False,
            )

        @self.mcp.tool()
        async def associate_or_dissociate_file(
            owner_type: str,
            owner_id: str,
            file_uris: Union[str, List[str]],
            mime_types: Optional[Union[str, List[str]]] = None,
            metadata: Optional[Union[Dict, List[Dict]]] = None,
            absolute_path: bool = True,
            operation: str = "associate",
            loader_kwargs: Optional[Dict[str, Any]] = None,
            long_sha: bool = False,
            preload_features: bool = False,
        ) -> Any:
            """Associate or dissociate files with a dataset or ensemble.

            Batch operations are supported by passing lists for file URIs and
            their matching metadata inputs.

            :param owner_type: ``"dataset"`` or ``"ensemble"``.
            :param owner_id: Dataset or ensemble ID or name.
            :param file_uris: Single file path or list of file paths.
            :param mime_types: MIME type or per-file MIME types. Required for
                ``operation="associate"``.
            :param metadata: Optional metadata mapping or per-file metadata
                mappings used for association.
            :param absolute_path: Apply Kosh path normalization to files that
                exist on disk.
            :param operation: ``"associate"`` or ``"dissociate"``.
            :param loader_kwargs: Optional loader keyword arguments stored with
                associated sources.
            :param long_sha: Compute and store a full content SHA for files
                that exist on disk.
            :param preload_features: Run ``list_features`` after association to
                populate the Kosh feature cache.
            :returns: A JSON-safe mapping describing the file association result.
            """
            return await self.run_tool(
                self.operations.associate_or_dissociate_file,
                owner_type,
                owner_id,
                file_uris,
                mime_types=mime_types,
                metadata=metadata,
                absolute_path=absolute_path,
                operation=operation,
                loader_kwargs=loader_kwargs,
                long_sha=long_sha,
                preload_features=preload_features,
                background=False,
            )

        @self.mcp.tool()
        async def update(
            object_type: str,
            object_id: str,
            attributes: Optional[Dict[str, Any]] = None,
            add_to_ensemble: Optional[str] = None,
            remove_from_ensemble: Optional[str] = None,
            inherit_attributes: bool = True,
            ensemble_tags: Optional[Dict[str, Any]] = None,
        ) -> Any:
            """Update datasets, ensembles, or source records.

            This can update attributes and manage dataset ensemble membership.

            :param object_type: ``"dataset"``, ``"ensemble"``, or ``"source"``.
            :param object_id: ID or name of the object to update.
            :param attributes: Attribute mapping to set or update.
            :param add_to_ensemble: Ensemble ID or name to add the dataset to.
            :param remove_from_ensemble: Ensemble ID or name to remove the
                dataset from.
            :param inherit_attributes: Whether added datasets inherit ensemble
                attributes.
            :param ensemble_tags: Optional tags to set for the dataset's
                ensemble membership.
            :returns: A JSON-safe mapping describing the update result.
            """
            return await self.run_tool(
                self.operations.update,
                object_type,
                object_id,
                attributes=attributes,
                add_to_ensemble=add_to_ensemble,
                remove_from_ensemble=remove_from_ensemble,
                inherit_attributes=inherit_attributes,
                ensemble_tags=ensemble_tags,
                background=False,
            )

        @self.mcp.tool()
        async def load(
            dataset_id: str,
            feature_name: Optional[Union[str, List[str]]] = None,
            action: str = "get",
            source_id: Optional[str] = None,
            output_format: Optional[str] = None,
            group: bool = False,
            use_cache: bool = True,
            verbose: bool = False,
            options: Optional[Dict[str, Any]] = None,
            list_features: bool = False,
        ) -> Any:
            """List, describe, or load feature data from a dataset.

            Use ``action="list_features"`` or ``list_features=True`` to inspect
            available features before retrieving data.

            :param dataset_id: Dataset ID or name.
            :param feature_name: Feature name or names for ``action="get"``, or
                a single feature name for ``action="describe_feature"``.
            :param action: ``"get"``, ``"list_features"``, or
                ``"describe_feature"``.
            :param source_id: Optional associated source ID that scopes the
                operation.
            :param output_format: Optional Kosh output format for
                ``action="get"``.
            :param group: Group multiple features in one ``action="get"`` call.
            :param use_cache: Use the Kosh feature cache for
                ``action="list_features"``.
            :param verbose: Return verbose loader feature discovery for
                ``action="list_features"``.
            :param options: Optional loader-specific keyword arguments.
            :param list_features: Backward-compatible alias for
                ``action="list_features"``.
            :returns: A JSON-safe mapping containing feature metadata or data.
            """
            return await self.run_tool(
                self.operations.load,
                dataset_id,
                feature_name=feature_name,
                action=action,
                source_id=source_id,
                output_format=output_format,
                group=group,
                use_cache=use_cache,
                verbose=verbose,
                options=options,
                list_features=list_features,
                background=False,
            )

        @self.mcp.tool()
        async def transform(
            action: str = "apply",
            inputs: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
            operator: Optional[Union[str, Dict[str, Any]]] = None,
            output_format: Optional[str] = None,
            options: Optional[Dict[str, Any]] = None,
        ) -> Any:
            """Apply Kosh transformers and operators to dataset feature graphs.

            Use ``action="list"`` to inspect the supported transformer and
            operator names.

            :param action: ``"apply"`` or ``"list"``.
            :param inputs: Input specification or specifications containing
                ``dataset_id``, ``feature_name``, and optional transformers.
            :param operator: Optional operator specification, currently
                ``KoshLNorm``.
            :param output_format: Optional Kosh output format for graph
                traversal.
            :param options: Optional traversal keyword arguments such as cache
                settings.
            :returns: A JSON-safe mapping containing transformed data or the
                transform catalog.
            """
            return await self.run_tool(
                self.operations.transform,
                action=action,
                inputs=inputs,
                operator=operator,
                output_format=output_format,
                options=options,
                background=False,
            )

        @self.mcp.tool()
        async def import_or_export(
            action: str,
            path: str,
            object_type: Optional[str] = None,
            object_ids: Optional[Union[str, List[str]]] = None,
            match_attributes: Optional[Union[str, List[str]]] = None,
            merge_handler: Optional[str] = None,
            overwrite: bool = False,
        ) -> Any:
            """Import native Kosh export files or export datasets and ensembles.

            :param action: ``"import"`` or ``"export"``.
            :param path: Input file for imports, or output file or directory for
                exports.
            :param object_type: ``"dataset"`` or ``"ensemble"``. Required for
                ``action="export"``.
            :param object_ids: Object ID or name, or a list of them, to export.
                Required for ``action="export"``.
            :param match_attributes: Attributes Kosh uses to match imported
                datasets or ensembles.
            :param merge_handler: Kosh merge handler name for imports.
            :param overwrite: Replace existing output files during export.
            :returns: A JSON-safe mapping describing the import or export result.
            """
            return await self.run_tool(
                self.operations.import_or_export,
                action,
                path,
                object_type=object_type,
                object_ids=object_ids,
                match_attributes=match_attributes,
                merge_handler=merge_handler,
                overwrite=overwrite,
                background=False,
            )

        @self.mcp.tool()
        async def dataframe(
            action: str,
            scope: str,
            object_id: Optional[str] = None,
            output_path: Optional[str] = None,
            output_format: str = "csv",
            data_columns: Optional[Union[str, List[str]]] = None,
            data_criteria: Optional[Dict[str, Any]] = None,
            ensemble_tags: Optional[Dict[str, Any]] = None,
            include_ensemble_attributes: bool = True,
            include_ensemble_tags: bool = True,
            overwrite: bool = False,
        ) -> Any:
            """Preview or save Kosh dataframe output.

            :param action: ``"preview"`` or ``"save"``.
            :param scope: ``"store"``, ``"dataset"``, or ``"ensemble"``.
            :param object_id: Dataset or ensemble ID or name for dataset and
                ensemble scopes.
            :param output_path: Output file or directory for ``action="save"``.
            :param output_format: ``"csv"`` or ``"json"`` for
                ``action="save"``.
            :param data_columns: Optional column or columns to include.
            :param data_criteria: Optional Kosh or Sina data filters.
            :param ensemble_tags: Optional ensemble tag filters for ensemble
                scope.
            :param include_ensemble_attributes: Include ensemble attributes for
                ensemble scope.
            :param include_ensemble_tags: Include ensemble tags for ensemble
                scope.
            :param overwrite: Replace an existing output file for
                ``action="save"``.
            :returns: A JSON-safe mapping describing the dataframe result.
            """
            return await self.run_tool(
                self.operations.dataframe,
                action,
                scope,
                object_id=object_id,
                output_path=output_path,
                output_format=output_format,
                data_columns=data_columns,
                data_criteria=data_criteria,
                ensemble_tags=ensemble_tags,
                include_ensemble_attributes=include_ensemble_attributes,
                include_ensemble_tags=include_ensemble_tags,
                overwrite=overwrite,
                background=False,
            )

        @self.mcp.tool()
        async def add_curve(
            dataset_id: str,
            name: str,
            x_values: List[float],
            y_values: List[float],
            x_label: Optional[str] = None,
            y_label: Optional[str] = None,
        ) -> Any:
            """Add paired 1D curve data directly to a dataset without files.

            This is useful for computed results, synthetic data, or quick data
            ingestion.

            :param dataset_id: Dataset ID.
            :param name: Name for the Kosh curve set and loadable feature.
            :param x_values: Independent curve values.
            :param y_values: Dependent curve values.
            :param x_label: Optional independent curve name. Defaults to
                ``"x"``.
            :param y_label: Optional dependent curve name. Defaults to
                ``"value"``.
            :returns: A JSON-safe mapping describing the created curve data.
            """
            return await self.run_tool(
                self.operations.add_curve,
                dataset_id,
                name,
                x_values,
                y_values,
                x_label=x_label,
                y_label=y_label,
                background=False,
            )

        @self.mcp.tool()
        async def delete(object_type: str, object_id: str) -> Any:
            """Delete a dataset or ensemble.

            This removes only the database entry. Associated files on disk are
            not deleted.

            :param object_type: ``"dataset"`` or ``"ensemble"``.
            :param object_id: ID or name of the dataset or ensemble to delete.
            :returns: A JSON-safe mapping describing the deletion result.
            """
            return await self.run_tool(
                self.operations.delete,
                object_type,
                object_id,
                background=False,
            )


def _create_kosh_server_class():
    """Create the concrete server class once mada-tools is available."""
    base_server = _load_base_server()

    class KoshServer(_KoshServerMixin, base_server):
        pass

    return KoshServer


class KoshServer:
    """Factory proxy for constructing the concrete Kosh MCP server."""

    def __new__(cls, *args, **kwargs):
        server_class = _create_kosh_server_class()
        return server_class(*args, **kwargs)


def main():
    """Run the Kosh MCP server."""
    try:
        server = KoshServer()
        server.run_with_args(server_key="kosh")
    except KoshMCPDependencyError as e:
        print(str(e), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
