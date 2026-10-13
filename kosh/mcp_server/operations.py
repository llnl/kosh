"""Business logic for the Kosh MCP server.

This module keeps Kosh operations independent from FastMCP registration so the
MCP adapter can delegate through ``BaseMCPServer.run_tool()``.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .serializers import KoshSerializer
from .store import KoshStoreHelper
from .transforms import (
    DEFAULT_OPERATOR_REGISTRY,
    DEFAULT_TRANSFORMER_REGISTRY,
    KoshTransformHelper,
)
from .utils import (
    as_list,
    check_json_object,
    check_path,
    coerce_file_uris,
    coerce_string_list,
    normalize_store_aliases,
    require_action,
    require_object_id,
    require_object_type,
    resolve_runtime_value,
    safe_filename,
)

# Keep the MCP-facing tool methods here and push repeated Kosh-specific
# concerns into helpers. This server has more structure than the simpler
# mada-tools servers because it spans store lookups, transforms, and mixed
# dataset/source/export workflows.


class KoshMCPOperations:
    """Encapsulate Kosh MCP tool implementations behind the run_tool contract."""

    DEFAULT_DATABASE_URI = "kosh_store.sqlite"
    DEFAULT_CONNECTION_TYPE = "write"
    DEFAULT_FIND_PAGE_SIZE = 20
    SUPPORTED_STORE_CONNECTION_ACTIONS = {"current", "connect"}
    TRANSFORMER_REGISTRY = DEFAULT_TRANSFORMER_REGISTRY
    OPERATOR_REGISTRY = DEFAULT_OPERATOR_REGISTRY

    def __init__(
        self,
        database_uri: str = DEFAULT_DATABASE_URI,
        connection_type: str = DEFAULT_CONNECTION_TYPE,
        execution_options: Optional[Dict[str, Any]] = None,
        store_aliases: Optional[Dict[str, str]] = None,
        serializer: Optional[KoshSerializer] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize Kosh operation helpers.

        :param database_uri: URI or path for the Kosh store.
        :param connection_type: Kosh connection type.
        :param execution_options: Optional execution options passed to
            ``kosh.connect``.
        :param store_aliases: Optional mapping of user-facing aliases to
            database URIs.
        :param serializer: Optional response serializer.
        :param logger: Optional logger.
        """
        self.database_uri = resolve_runtime_value(database_uri, self.DEFAULT_DATABASE_URI)
        self.connection_type = resolve_runtime_value(connection_type, self.DEFAULT_CONNECTION_TYPE)
        self.execution_options = check_json_object(execution_options, "execution_options")
        self.store_aliases = normalize_store_aliases(store_aliases)
        self.serializer = serializer or KoshSerializer()
        self.logger = logger or logging.getLogger("kosh_mcp_server")
        # These helpers keep the tool entrypoints close to Kosh API calls
        # instead of repeating store, transform, and serialization branches.
        self.store_helper = KoshStoreHelper(
            database_uri=self.database_uri,
            connection_type=self.connection_type,
            execution_options=self.execution_options,
            logger=self.logger,
        )
        self.transform_helper = KoshTransformHelper(
            serializer=self.serializer,
            resolve_object_id=self.store_helper.object_name_to_id,
            transformer_registry=self.TRANSFORMER_REGISTRY,
            operator_registry=self.OPERATOR_REGISTRY,
        )

    def configure(
        self,
        database_uri: Optional[str] = None,
        connection_type: Optional[str] = None,
        execution_options: Optional[Dict[str, Any]] = None,
        store_aliases: Optional[Dict[str, str]] = None,
    ) -> None:
        """Update runtime Kosh connection settings.

        :param database_uri: URI or path for the Kosh store.
        :param connection_type: Kosh connection type.
        :param execution_options: Optional execution options passed to
            ``kosh.connect``.
        :param store_aliases: Optional mapping of user-facing aliases to
            database URIs.
        """
        self.database_uri = resolve_runtime_value(database_uri, self.DEFAULT_DATABASE_URI)
        self.connection_type = resolve_runtime_value(connection_type, self.DEFAULT_CONNECTION_TYPE)
        self.execution_options = check_json_object(execution_options, "execution_options")
        self.store_aliases = normalize_store_aliases(store_aliases)
        self.store_helper.configure(
            database_uri=self.database_uri,
            connection_type=self.connection_type,
            execution_options=self.execution_options,
        )
        self.transform_helper.configure(
            serializer=self.serializer,
            transformer_registry=self.TRANSFORMER_REGISTRY,
            operator_registry=self.OPERATOR_REGISTRY,
        )

    def validate_connection(self) -> None:
        """Open and close the configured Kosh store to validate connectivity."""
        self.store_helper.validate_connection()

    def _connection_payload(
        self,
        action: str,
        selection_type: str,
        alias: Optional[str] = None,
        store_existed_before_connect: Optional[bool] = None,
        created_new_store: bool = False,
        warnings: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Build the MCP response payload for the active Kosh store.

        :param action: Normalized connection action name.
        :param selection_type: How the active store was selected.
        :param alias: Alias used to select the active store, if any.
        :param store_existed_before_connect: Whether the target store existed
            before the connect attempt.
        :param created_new_store: Whether the connect attempt created a new
            local store file.
        :param warnings: Optional warning messages for the response payload.
        :returns: JSON-safe mapping describing the active store connection.
        """
        connection_details = self.store_helper.describe_connection()
        resolved_alias = alias
        if resolved_alias is None:
            matches = [
                configured_alias
                for configured_alias, configured_database_uri in self.store_aliases.items()
                if configured_database_uri == self.database_uri
            ]
            if len(matches) == 1:
                resolved_alias = matches[0]
        return {
            "status": "connected",
            "action": action,
            "selection_type": selection_type,
            "alias": resolved_alias,
            "database_uri": connection_details["database_uri"],
            "full_path": connection_details["full_path"],
            "connection_type": self.connection_type,
            "store_exists": connection_details["store_exists"],
            "store_existed_before_connect": store_existed_before_connect,
            "created_new_store": created_new_store,
            "warnings": warnings or [],
        }

    def store_connection(
        self,
        action: str = "current",
        alias: Optional[str] = None,
        database_uri: Optional[str] = None,
    ) -> Tuple[bool, Any]:
        """Inspect or switch the active Kosh store used by subsequent tool calls.

        Use ``action="current"`` to inspect the active store and
        ``action="connect"`` to switch stores. The current
        ``connection_type`` and ``execution_options`` remain unchanged.

        :param action: ``"current"`` or ``"connect"``.
        :param alias: Preconfigured store alias for ``action="connect"``.
        :param database_uri: Raw Kosh store URI or local path for
            ``action="connect"``.
        :returns: Tuple of a success flag and a JSON-safe payload or error
            string.
        """
        previous_settings = {
            "database_uri": self.database_uri,
            "connection_type": self.connection_type,
            "execution_options": dict(self.execution_options),
            "store_aliases": dict(self.store_aliases),
        }
        configured_target = False

        try:
            resolved_action = require_action(
                action,
                self.SUPPORTED_STORE_CONNECTION_ACTIONS,
                param_name="action",
            )
            has_alias = isinstance(alias, str) and bool(alias.strip())
            has_database_uri = (
                isinstance(database_uri, str) and bool(database_uri.strip())
            )
            if resolved_action == "current":
                if has_alias or has_database_uri:
                    raise ValueError(
                        "store_connection(action='current') does not accept alias or database_uri."
                    )
                return True, self._connection_payload(
                    action=resolved_action,
                    selection_type="current",
                )
            if has_alias == has_database_uri:
                raise ValueError(
                    "store_connection(action='connect') requires exactly one of alias or database_uri."
                )

            resolved_alias = None
            if has_alias:
                resolved_alias = alias.strip()
                if resolved_alias not in self.store_aliases:
                    if self.store_aliases:
                        available_aliases = ", ".join(sorted(self.store_aliases))
                        raise ValueError(
                            f"Unknown store alias '{resolved_alias}'. "
                            f"Available store aliases: {available_aliases}."
                        )
                    raise ValueError(
                        f"Unknown store alias '{resolved_alias}'. "
                        "No store aliases are configured."
                    )
                resolved_database_uri = self.store_aliases[resolved_alias]
                selection_type = "alias"
            else:
                resolved_database_uri = database_uri.strip()
                selection_type = "database_uri"

            connection_details = self.store_helper.describe_database_uri(
                resolved_database_uri
            )
            local_store_path = (
                Path(connection_details["full_path"])
                if connection_details["full_path"] is not None
                else None
            )
            store_existed_before_connect = connection_details["store_exists"]

            self.configure(
                database_uri=resolved_database_uri,
                connection_type=self.connection_type,
                execution_options=self.execution_options,
                store_aliases=self.store_aliases,
            )
            configured_target = True
            self.validate_connection()

            warnings = []
            created_new_store = False
            if local_store_path is not None:
                created_new_store = (
                    not store_existed_before_connect and local_store_path.exists()
                )
                if created_new_store:
                    warnings.append(
                        f"Local store path '{resolved_database_uri}' did not exist and was created."
                    )

            return True, self._connection_payload(
                action=resolved_action,
                selection_type=selection_type,
                alias=resolved_alias,
                store_existed_before_connect=store_existed_before_connect,
                created_new_store=created_new_store,
                warnings=warnings,
            )
        except Exception as e:
            if configured_target:
                self.configure(**previous_settings)
            return False, str(e)

    def connect_store(
        self,
        alias: Optional[str] = None,
        database_uri: Optional[str] = None,
    ) -> Tuple[bool, Any]:
        """Dispatch the legacy store connection helper through ``store_connection``.

        :param alias: Preconfigured store alias.
        :param database_uri: Raw Kosh store URI or local path.
        :returns: Tuple of a success flag and a JSON-safe payload or error
            string.
        """
        action = "connect" if alias or database_uri else "current"
        return self.store_connection(
            action=action,
            alias=alias,
            database_uri=database_uri,
        )

    def _validate_association_args(
        self,
        file_uris: Union[str, List[str]],
        mime_types: Optional[Union[str, List[str]]],
        metadata: Optional[Union[Dict, List[Dict]]] = None,
        loader_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[str], Union[str, List[str]], Optional[Union[Dict, List[Dict]]]]:
        """Validate association arguments before delegating to Kosh.

        :param file_uris: File URI or URIs to associate.
        :param mime_types: MIME type or MIME types for the files.
        :param metadata: Optional metadata mapping or per-file metadata.
        :param loader_kwargs: Optional loader keyword arguments stored on the
            association.
        :returns: Tuple of normalized file URIs, MIME types, and metadata.
        """
        uris = coerce_file_uris(file_uris)

        if not mime_types:
            raise ValueError(
                "associate_or_dissociate_file requires mime_types because Kosh associate() requires mime_type."
            )

        if not isinstance(mime_types, str):
            try:
                mimes = list(mime_types)
            except TypeError as e:
                raise ValueError(
                    "associate_or_dissociate_file requires mime_types to be a string or list of strings."
                ) from e
            if len(mimes) != len(uris):
                raise ValueError(
                    f"associate_or_dissociate_file received {len(mimes)} MIME type(s) for {len(uris)} file URI(s)."
                )
            if any(not isinstance(mime, str) or not mime.strip() for mime in mimes):
                raise ValueError("associate_or_dissociate_file MIME types must be non-empty strings.")

        if metadata is not None and not isinstance(metadata, dict):
            try:
                metas = list(metadata)
            except TypeError as e:
                raise ValueError(
                    "associate_or_dissociate_file metadata must be a JSON object or list of JSON objects."
                ) from e
            if len(metas) != len(uris):
                raise ValueError(
                    f"associate_or_dissociate_file received {len(metas)} metadata item(s) for {len(uris)} file URI(s)."
                )
            if any(not isinstance(meta, dict) for meta in metas):
                raise ValueError("associate_or_dissociate_file metadata entries must be dictionaries.")

        if loader_kwargs is not None:
            if not isinstance(loader_kwargs, dict):
                raise ValueError("associate_or_dissociate_file loader_kwargs must be a JSON object.")
            if metadata is not None and not isinstance(metadata, dict):
                raise ValueError(
                    "associate_or_dissociate_file loader_kwargs cannot be combined with per-file metadata."
                )

        return uris, mime_types, metadata

    def find(
        self,
        object_type: str,
        ids: Optional[Union[str, List[str]]] = None,
        data_criteria: Optional[Dict[str, Any]] = None,
        ensemble_id: Optional[str] = None,
        file_uri: Optional[str] = None,
        owner_type: Optional[str] = None,
        owner_id: Optional[str] = None,
        mode: str = "auto",
        page_size: int = DEFAULT_FIND_PAGE_SIZE,
        detail: str = "compact",
        allow_expensive: bool = False,
    ) -> Tuple[bool, Any]:
        """Query datasets, ensembles, or source/file records.

        :param object_type: ``"dataset"``, ``"ensemble"``, or ``"source"``.
        :param ids: Optional ID or IDs, or names for dataset and ensemble
            queries.
        :param data_criteria: Optional Kosh or Sina attribute filters.
        :param ensemble_id: Optional ensemble ID or name for dataset queries.
        :param file_uri: Optional associated file URI filter.
        :param owner_type: Optional owner type for source queries.
        :param owner_id: Optional owner ID or name for source queries.
        :param mode: ``"auto"``, ``"summary"``, ``"count"``, ``"ids"``, or
            ``"objects"``.
        :param page_size: Maximum number of results to return for non-count
            modes.
        :param detail: ``"compact"`` or ``"standard"`` object summaries.
        :param allow_expensive: Opt in to full Kosh/Sina scans when no fast
            path is available.
        :returns: Tuple of a success flag and a JSON-safe payload or error
            string.
        """
        try:
            obj_type = require_object_type(object_type, allow_source=True)
            requested_mode = require_action(
                mode,
                {"auto", "summary", "count", "ids", "objects"},
                param_name="mode",
            )
            normalized_detail = require_action(
                detail,
                {"compact", "standard"},
                param_name="detail",
            )
            if page_size is None:
                normalized_page_size = self.DEFAULT_FIND_PAGE_SIZE
            else:
                try:
                    normalized_page_size = int(page_size)
                except (TypeError, ValueError) as e:
                    raise ValueError("page_size must be a non-negative integer.") from e
                if normalized_page_size < 0:
                    raise ValueError("page_size must be a non-negative integer.")
            with self.store_helper.connection() as store:
                resolved_ids = None
                if ids:
                    raw_ids = coerce_string_list(ids, "ids")
                    if obj_type == "source":
                        resolved_ids = raw_ids
                    else:
                        resolved_ids = [
                            self.store_helper.object_name_to_id(store, obj_ref, obj_type)
                            for obj_ref in raw_ids
                        ]
                resolved_ensemble_id = (
                    self.store_helper.object_name_to_id(store, ensemble_id, "ensemble")
                    if ensemble_id else None
                )
                broad_query = not any(
                    (
                        resolved_ids,
                        data_criteria,
                        resolved_ensemble_id,
                        file_uri,
                        owner_type,
                        owner_id,
                    )
                )
                if obj_type == "source":
                    if owner_id or owner_type:
                        query_case = "source_by_owner"
                    elif resolved_ids:
                        query_case = "object_lookup"
                    else:
                        query_case = "source_store_query"
                elif resolved_ids:
                    query_case = "object_lookup"
                elif resolved_ensemble_id:
                    query_case = "ensemble_membership"
                elif obj_type == "ensemble":
                    query_case = "store_ensembles"
                else:
                    query_case = "store_datasets"

                effective_mode = requested_mode
                if requested_mode == "auto":
                    effective_mode = "summary" if broad_query else "objects"

                fast_page = self.store_helper.fast_find_page(
                    store,
                    object_type=obj_type,
                    ids=resolved_ids,
                    data_criteria=data_criteria,
                    ensemble_id=resolved_ensemble_id,
                    file_uri=file_uri,
                    owner_type=owner_type,
                    owner_id=owner_id,
                    page_size=0 if effective_mode == "count" else normalized_page_size,
                )
                if fast_page is not None:
                    payload = {
                        "object_type": obj_type,
                        "query_case": query_case,
                        "mode": effective_mode,
                        "execution_path": "sql_fast_path",
                    }
                    if requested_mode != effective_mode:
                        payload["requested_mode"] = requested_mode
                    payload.update(
                        {
                            "count": fast_page["count"],
                            "count_exact": True,
                            "count_method": fast_page["count_method"],
                        }
                    )
                    if effective_mode == "count":
                        return True, payload

                    if effective_mode == "ids":
                        results = [object_id for _, object_id in fast_page["handles"]]
                    else:
                        objects = [
                            query_store.open(object_id)
                            for query_store, object_id in fast_page["handles"]
                        ]
                        results = [
                            self.serializer.serialize_find_object(
                                obj,
                                detail=normalized_detail,
                            )
                            for obj in objects
                        ]
                        payload["detail"] = normalized_detail

                    payload.update(
                        {
                            "returned": len(results),
                            "page_size": normalized_page_size,
                            "truncated": fast_page["truncated"],
                            "results": results,
                        }
                    )
                    return True, payload

                if effective_mode == "count" and not allow_expensive and not resolved_ids:
                    payload = {
                        "object_type": obj_type,
                        "query_case": query_case,
                        "mode": effective_mode,
                        "execution_path": "blocked_expensive_scan",
                        "status": "requires_expensive_scan",
                        "reason": (
                            "Exact counts for this query require a full Kosh/Sina scan. "
                            "Retry with allow_expensive=True or narrow the query."
                        ),
                        "allow_expensive": False,
                    }
                    if requested_mode != effective_mode:
                        payload["requested_mode"] = requested_mode
                    return True, payload

                if effective_mode == "summary" and not allow_expensive and not resolved_ids:
                    payload = {
                        "object_type": obj_type,
                        "query_case": query_case,
                        "mode": effective_mode,
                        "execution_path": "blocked_expensive_scan",
                        "status": "requires_expensive_scan",
                        "reason": (
                            "Summary mode needs an exact count for this query and would "
                            "require a full Kosh/Sina scan. Retry with allow_expensive=True "
                            "or use mode='objects'."
                        ),
                        "allow_expensive": False,
                    }
                    if requested_mode != effective_mode:
                        payload["requested_mode"] = requested_mode
                    return True, payload

                if (
                    effective_mode in {"summary", "ids", "objects"}
                    and not allow_expensive
                    and broad_query
                ):
                    payload = {
                        "object_type": obj_type,
                        "query_case": query_case,
                        "mode": effective_mode,
                        "execution_path": "blocked_expensive_scan",
                        "status": "requires_expensive_scan",
                        "reason": (
                            "This broad query would require a full Kosh/Sina scan on this "
                            "backend. Narrow the query or retry with allow_expensive=True."
                        ),
                        "allow_expensive": False,
                    }
                    if requested_mode != effective_mode:
                        payload["requested_mode"] = requested_mode
                    return True, payload

                payload = {
                    "object_type": obj_type,
                    "query_case": query_case,
                    "mode": effective_mode,
                    "execution_path": "kosh_find",
                }
                if requested_mode != effective_mode:
                    payload["requested_mode"] = requested_mode

                if effective_mode == "count":
                    total_count = self.store_helper.count_objects(
                        store,
                        object_type=obj_type,
                        ids=resolved_ids,
                        data_criteria=data_criteria,
                        ensemble_id=resolved_ensemble_id,
                        file_uri=file_uri,
                        owner_type=owner_type,
                        owner_id=owner_id,
                    )
                    payload.update(
                        {
                            "count": total_count,
                            "count_exact": True,
                            "count_method": "kosh_scan",
                        }
                    )
                    return True, payload

                if effective_mode == "ids":
                    result_ids, truncated = self.store_helper.find_object_ids(
                        store,
                        object_type=obj_type,
                        ids=resolved_ids,
                        data_criteria=data_criteria,
                        ensemble_id=resolved_ensemble_id,
                        file_uri=file_uri,
                        owner_type=owner_type,
                        owner_id=owner_id,
                        limit=normalized_page_size,
                    )
                    payload.update(
                        {
                            "returned": len(result_ids),
                            "page_size": normalized_page_size,
                            "truncated": truncated,
                            "results": result_ids,
                        }
                    )
                    return True, payload

                results, truncated = self.store_helper.find_objects(
                    store,
                    object_type=obj_type,
                    ids=resolved_ids,
                    data_criteria=data_criteria,
                    ensemble_id=resolved_ensemble_id,
                    file_uri=file_uri,
                    owner_type=owner_type,
                    owner_id=owner_id,
                    limit=normalized_page_size,
                )
                serialized = [
                    self.serializer.serialize_find_object(
                        obj,
                        detail=normalized_detail,
                    )
                    for obj in results
                ]
                payload["detail"] = normalized_detail
                payload.update(
                    {
                        "returned": len(serialized),
                        "page_size": normalized_page_size,
                        "truncated": truncated,
                        "results": serialized,
                    }
                )
                if effective_mode == "summary":
                    total_count = self.store_helper.count_objects(
                        store,
                        object_type=obj_type,
                        ids=resolved_ids,
                        data_criteria=data_criteria,
                        ensemble_id=resolved_ensemble_id,
                        file_uri=file_uri,
                        owner_type=owner_type,
                        owner_id=owner_id,
                    )
                    payload.update(
                        {
                            "count": total_count,
                            "count_exact": True,
                            "count_method": "kosh_scan",
                            "truncated": total_count > len(serialized),
                        }
                    )

                return True, payload
        except Exception as e:
            return False, str(e)

    def create_dataset_or_ensemble(
        self,
        object_type: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Any]:
        """Create a dataset or ensemble.

        :param object_type: ``"dataset"`` or ``"ensemble"``.
        :param name: Optional object name.
        :param metadata: Optional metadata attributes for the new object.
        :returns: Tuple of a success flag and a JSON-safe payload or error
            string.
        """
        try:
            with self.store_helper.connection() as store:
                obj_type = require_object_type(object_type)
                if metadata is not None and not isinstance(metadata, dict):
                    raise ValueError("metadata must be a JSON object.")
                attributes = metadata or {}
                create_args = {"metadata": attributes}
                if name is not None:
                    create_args["name"] = name

                if obj_type == "dataset":
                    obj = store.create(**create_args)
                elif obj_type == "ensemble":
                    obj = store.create_ensemble(**create_args)
                else:
                    raise ValueError(f"Unknown object type: {object_type}")

                details = self.serializer.serialize_object(
                    obj,
                    include_files=False,
                )

                return True, details
        except Exception as e:
            return False, str(e)

    def associate_or_dissociate_file(
        self,
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
    ) -> Tuple[bool, Any]:
        """Associate or dissociate files with a dataset or ensemble.

        :param owner_type: ``"dataset"`` or ``"ensemble"``.
        :param owner_id: Dataset or ensemble ID or name.
        :param file_uris: File URI or URIs to process.
        :param mime_types: MIME type or MIME types for association.
        :param metadata: Optional metadata mapping or per-file metadata.
        :param absolute_path: Normalize file paths before Kosh association.
        :param operation: ``"associate"`` or ``"dissociate"``.
        :param loader_kwargs: Optional loader keyword arguments stored on the
            association.
        :param long_sha: Compute a full SHA for files that exist on disk.
        :param preload_features: Preload feature metadata after association.
        :returns: Tuple of a success flag and a JSON-safe payload or error
            string.
        """
        try:
            with self.store_helper.connection() as store:
                obj_type = require_object_type(owner_type)
                owner_id = self.store_helper.object_name_to_id(store, owner_id, obj_type)
                owner = store.open(owner_id)
                if not isinstance(operation, str):
                    raise ValueError("operation must be 'associate' or 'dissociate'.")
                normalized_operation = operation.lower().strip()

                if normalized_operation in {"dissociate", "remove", "unassociate"}:
                    uris = coerce_file_uris(file_uris)
                    for uri in uris:
                        owner.dissociate(uri, absolute_path=absolute_path)

                    return True, {
                        "owner_id": owner_id,
                        "operation": "dissociate",
                        "file_count": len(uris),
                        "files": uris,
                    }

                if normalized_operation != "associate":
                    raise ValueError("operation must be 'associate' or 'dissociate'.")

                if not isinstance(long_sha, bool):
                    raise ValueError("long_sha must be a boolean.")
                if not isinstance(preload_features, bool):
                    raise ValueError("preload_features must be a boolean.")

                uris, mimes, metas = self._validate_association_args(
                    file_uris,
                    mime_types,
                    metadata,
                    loader_kwargs=loader_kwargs,
                )
                source_ids = owner.associate(
                    uris,
                    mime_type=mimes,
                    metadata=metas or {},
                    absolute_path=absolute_path,
                    loader_kwargs=loader_kwargs,
                    long_sha=long_sha,
                    preload_features=preload_features,
                    id_only=True,
                )
                return True, {
                    "owner_id": owner_id,
                    "operation": "associate",
                    "file_count": len(uris),
                    "files": uris,
                    "mime_types": mimes,
                    "source_ids": [str(source_id) for source_id in as_list(source_ids)],
                    "loader_kwargs_applied": loader_kwargs is not None,
                    "long_sha": long_sha,
                    "preload_features": preload_features,
                }
        except Exception as e:
            return False, str(e)

    def update(
        self,
        object_type: str,
        object_id: str,
        attributes: Optional[Dict[str, Any]] = None,
        add_to_ensemble: Optional[str] = None,
        remove_from_ensemble: Optional[str] = None,
        inherit_attributes: bool = True,
        ensemble_tags: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Any]:
        """Update object attributes and ensemble membership.

        :param object_type: ``"dataset"``, ``"ensemble"``, or ``"source"``.
        :param object_id: Object ID or name.
        :param attributes: Optional attribute mapping to update.
        :param add_to_ensemble: Optional ensemble ID or name to add a dataset
            to.
        :param remove_from_ensemble: Optional ensemble ID or name to remove a
            dataset from.
        :param inherit_attributes: Apply Kosh ensemble attribute inheritance
            when adding a dataset.
        :param ensemble_tags: Optional ensemble tags used when adding a
            dataset.
        :returns: Tuple of a success flag and a JSON-safe payload or error
            string.
        """
        try:
            with self.store_helper.connection() as store:
                obj_type = require_object_type(object_type, allow_source=True)
                operations = []

                if attributes is not None and not isinstance(attributes, dict):
                    raise ValueError("attributes must be a JSON object.")

                if obj_type == "source":
                    source_id = require_object_id("object_id", object_id)
                    membership_options = (
                        add_to_ensemble
                        or remove_from_ensemble
                        or ensemble_tags is not None
                        or inherit_attributes is not True
                    )
                    if membership_options:
                        raise ValueError("Ensemble membership options are only supported for dataset updates.")

                    blocked = {"uri", "mime_type", "associated"}.intersection(attributes or {})
                    if blocked:
                        blocked_text = ", ".join(sorted(blocked))
                        raise ValueError(f"Source identity fields cannot be updated: {blocked_text}.")

                    obj = store._load(source_id)
                    if not self.store_helper.object_matches_type(obj, "source"):
                        raise ValueError(f"Source '{source_id}' was not found.")
                    if attributes:
                        obj.update(attributes)
                        operations.append(f"updated {len(attributes)} attribute(s)")
                    return True, {
                        "operations": operations,
                        "object": self.serializer.serialize_source(obj),
                    }

                object_id = self.store_helper.object_name_to_id(store, object_id, obj_type)
                obj = store.open(object_id)

                if (add_to_ensemble or remove_from_ensemble) and obj_type != "dataset":
                    raise ValueError("Ensemble membership updates can only be applied to datasets.")

                if not isinstance(inherit_attributes, bool):
                    raise ValueError("inherit_attributes must be a boolean.")
                if ensemble_tags is not None:
                    if not isinstance(ensemble_tags, dict):
                        raise ValueError("ensemble_tags must be a JSON object.")
                    if not add_to_ensemble:
                        raise ValueError("ensemble_tags can only be provided when add_to_ensemble is set.")

                if attributes:
                    obj.update(attributes)
                    operations.append(f"updated {len(attributes)} attribute(s)")

                if add_to_ensemble:
                    ensemble_id = self.store_helper.object_name_to_id(store, add_to_ensemble, "ensemble")
                    ensemble = store.open(ensemble_id)
                    ensemble.add(
                        obj,
                        inherit_attributes=inherit_attributes,
                        ensemble_tags=ensemble_tags,
                    )
                    operations.append(f"added to ensemble '{ensemble_id}'")

                if remove_from_ensemble:
                    ensemble_id = self.store_helper.object_name_to_id(
                        store,
                        remove_from_ensemble,
                        "ensemble",
                    )
                    ensemble = store.open(ensemble_id)
                    ensemble.remove(obj)
                    operations.append(f"removed from ensemble '{ensemble_id}'")

                return True, {
                    "operations": operations,
                    "object": self.serializer.serialize_object(
                        obj,
                        include_files=False,
                    ),
                }
        except Exception as e:
            return False, str(e)

    def load(
        self,
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
    ) -> Tuple[bool, Any]:
        """List, describe, or load feature data from a dataset.

        :param dataset_id: Dataset ID or name.
        :param feature_name: Feature name or names for load operations.
        :param action: ``"get"``, ``"list_features"``, or
            ``"describe_feature"``.
        :param source_id: Optional associated source ID.
        :param output_format: Optional Kosh output format for ``action="get"``.
        :param group: Group multiple features in one ``get`` call.
        :param use_cache: Use the Kosh feature cache.
        :param verbose: Return verbose feature discovery results.
        :param options: Optional loader-specific keyword arguments.
        :param list_features: Backward-compatible alias for
            ``action="list_features"``.
        :returns: Tuple of a success flag and a JSON-safe payload or error
            string.
        """
        try:
            load_options = check_json_object(options, "options")
            if not all(isinstance(value, bool) for value in (group, use_cache, verbose, list_features)):
                raise ValueError("group, use_cache, verbose, and list_features must be booleans.")

            normalized_action = require_action(
                action,
                {"get", "list_features", "describe_feature"},
            )
            if list_features:
                if normalized_action not in {"get", "list_features"}:
                    raise ValueError("list_features=True is only compatible with action='list_features'.")
                normalized_action = "list_features"

            with self.store_helper.connection() as store:
                dataset_id = self.store_helper.object_name_to_id(store, dataset_id, "dataset")
                dataset = store.open(dataset_id)
                resolved_source_id = (
                    require_object_id("source_id", source_id)
                    if source_id else None
                )

                if normalized_action == "list_features":
                    features = dataset.list_features(
                        Id=resolved_source_id,
                        use_cache=use_cache,
                        verbose=verbose,
                        **load_options,
                    )
                    return True, {
                        "action": normalized_action,
                        "dataset_id": dataset_id,
                        "source_id": resolved_source_id,
                        "feature_count": len(features),
                        "features": list(features),
                    }

                if not feature_name:
                    raise ValueError("Provide feature_name to load or describe, or use action='list_features'.")

                if normalized_action == "describe_feature":
                    if not isinstance(feature_name, str):
                        raise ValueError("describe_feature requires a single feature_name string.")
                    description = dataset.describe_feature(
                        feature_name,
                        Id=resolved_source_id,
                        **load_options,
                    )
                    return True, {
                        "action": normalized_action,
                        "dataset_id": dataset_id,
                        "source_id": resolved_source_id,
                        "feature_name": feature_name,
                        "description": self.serializer.serialize(description),
                    }

                data = dataset.get(
                    feature_name,
                    Id=resolved_source_id,
                    format=output_format,
                    group=group,
                    **load_options,
                )
                serialized = self.serializer.serialize(data)

                return True, {
                    "action": normalized_action,
                    "dataset_id": dataset_id,
                    "source_id": resolved_source_id,
                    "feature_name": feature_name,
                    "output_format": output_format,
                    "group": group,
                    "data": serialized,
                }
        except Exception as e:
            return False, str(e)

    def transform(
        self,
        action: str = "apply",
        inputs: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        operator: Optional[Union[str, Dict[str, Any]]] = None,
        output_format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Any]:
        """Apply allow-listed Kosh transformers and operators to feature graphs.

        :param action: ``"apply"`` or ``"list"``.
        :param inputs: Transform input specification or specifications.
        :param operator: Optional operator specification applied to the input
            graphs.
        :param output_format: Optional Kosh traversal output format.
        :param options: Optional traversal keyword arguments.
        :returns: Tuple of a success flag and a JSON-safe payload or error
            string.
        """
        try:
            normalized_action = require_action(action, {"apply", "list"})
            if normalized_action == "list":
                return True, {
                    "action": "list",
                    "transformers": self.transform_helper.catalog_entries(
                        self.TRANSFORMER_REGISTRY,
                        "transformer",
                    ),
                    "operators": self.transform_helper.catalog_entries(
                        self.OPERATOR_REGISTRY,
                        "operator",
                    ),
                }

            execution_options = check_json_object(options, "options")
            input_specs = self.transform_helper.normalize_transform_inputs(inputs)
            with self.store_helper.connection() as store:
                graph_items = [
                    self.transform_helper.transform_input_graph(store, input_spec)
                    for input_spec in input_specs
                ]
                graphs = [graph for graph, _ in graph_items]
                input_summaries = [summary for _, summary in graph_items]

                if operator is not None:
                    operator_name, operator_class, operator_options = (
                        self.transform_helper.normalize_transform_spec(
                            operator,
                            self.OPERATOR_REGISTRY,
                            "operator",
                        )
                    )
                    result_graph = operator_class(*graphs, **operator_options)
                    data = result_graph.traverse(format=output_format, **execution_options)
                    operator_summary = {
                        "name": operator_name,
                        "options": operator_options,
                    }
                else:
                    results = [
                        graph.traverse(format=output_format, **execution_options)
                        for graph in graphs
                    ]
                    data = results[0] if len(results) == 1 else results
                    operator_summary = None

                payload = {
                    "action": normalized_action,
                    "inputs": input_summaries,
                    "operator": operator_summary,
                    "output_format": output_format,
                    "data": self.serializer.serialize(data),
                }
                return True, {
                    key: value for key, value in payload.items() if value is not None
                }
        except Exception as e:
            return False, str(e)

    def _prepare_write_file(
        self,
        path: str,
        default_filename: str,
        overwrite: bool,
    ) -> Path:
        """Prepare a concrete file path for tool output.

        :param path: Requested output path.
        :param default_filename: Filename to append when ``path`` is a
            directory.
        :param overwrite: Replace an existing file at the resolved path.
        :returns: Concrete writable output path.
        """
        output_path = check_path(path)
        if output_path.exists() and output_path.is_dir():
            output_path = output_path / default_filename
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            if output_path.is_dir():
                raise ValueError(f"Output path '{output_path}' is a directory.")
            if not overwrite:
                raise ValueError(
                    f"Output path '{output_path}' already exists. Set overwrite=True to replace it."
                )
        return output_path

    def _prepare_export_paths(
        self,
        path: str,
        object_type: str,
        objects: List[Any],
        overwrite: bool,
    ) -> List[Path]:
        """Return concrete native export paths for one or more objects.

        :param path: Requested export path.
        :param object_type: Canonical object type being exported.
        :param objects: Objects that will be exported.
        :param overwrite: Replace existing export files.
        :returns: Concrete export paths for the objects.
        """
        base_path = check_path(path)
        if len(objects) > 1:
            if base_path.exists() and not base_path.is_dir():
                raise ValueError(
                    "Exporting multiple objects requires path to be a directory."
                )
            base_path.mkdir(parents=True, exist_ok=True)
            paths = [
                base_path / f"{object_type}_{safe_filename(str(obj.id))}.json"
                for obj in objects
            ]
        else:
            obj = objects[0]
            default_filename = f"{object_type}_{safe_filename(str(obj.id))}.json"
            paths = [self._prepare_write_file(str(base_path), default_filename, overwrite)]

        for export_path in paths:
            if export_path.exists():
                if export_path.is_dir():
                    raise ValueError(f"Export path '{export_path}' is a directory.")
                if not overwrite:
                    raise ValueError(
                        f"Export path '{export_path}' already exists. Set overwrite=True to replace it."
                    )
                export_path.unlink()
        return paths

    def import_or_export(
        self,
        action: str,
        path: str,
        object_type: Optional[str] = None,
        object_ids: Optional[Union[str, List[str]]] = None,
        match_attributes: Optional[Union[str, List[str]]] = None,
        merge_handler: Optional[str] = None,
        overwrite: bool = False,
    ) -> Tuple[bool, Any]:
        """Import native Kosh exports or export datasets/ensembles to files.

        :param action: ``"import"`` or ``"export"``.
        :param path: Import file path or export file or directory path.
        :param object_type: ``"dataset"`` or ``"ensemble"`` for exports.
        :param object_ids: Object ID or IDs, or names, to export.
        :param match_attributes: Attributes Kosh uses to match imported
            objects.
        :param merge_handler: Merge policy name for imports.
        :param overwrite: Replace existing export files.
        :returns: Tuple of a success flag and a JSON-safe payload or error
            string.
        """
        try:
            normalized_action = require_action(action, {"import", "export"})
            with self.store_helper.connection() as store:
                if normalized_action == "export":
                    obj_type = require_object_type(object_type or "")
                    object_refs = coerce_string_list(
                        object_ids,
                        "object_ids",
                        required=True,
                    )
                    objects = [
                        store.open(self.store_helper.object_name_to_id(store, obj_ref, obj_type))
                        for obj_ref in object_refs
                    ]
                    export_paths = self._prepare_export_paths(path, obj_type, objects, overwrite)
                    for obj, export_path in zip(objects, export_paths):
                        obj.export(file=str(export_path))

                    return True, {
                        "count": len(objects),
                        "paths": [str(export_path) for export_path in export_paths],
                        "object_ids": [obj.id for obj in objects],
                    }

                input_path = check_path(path)
                if input_path.is_dir():
                    raise ValueError("Import path must be a file, not a directory.")
                if not input_path.exists():
                    raise ValueError(f"Import path '{input_path}' does not exist.")
                attributes = coerce_string_list(
                    match_attributes if match_attributes is not None else ["name"],
                    "match_attributes",
                    required=True,
                )
                if merge_handler is not None:
                    if not isinstance(merge_handler, str) or not merge_handler.strip():
                        raise ValueError("merge_handler must be a non-empty string when provided.")
                    normalized_merge_handler = merge_handler.strip()
                    if normalized_merge_handler not in {"conservative", "overwrite", "preserve"}:
                        raise ValueError(
                            "merge_handler must be 'conservative', 'overwrite', or 'preserve'."
                        )
                else:
                    normalized_merge_handler = None

                imported = store.import_dataset(
                    str(input_path),
                    match_attributes=attributes,
                    merge_handler=normalized_merge_handler,
                )
                imported_objects = as_list(imported)
                serialized = [
                    self.serializer.serialize_object(
                        obj,
                        include_files=False,
                    )
                    for obj in imported_objects
                ]

                return True, {
                    "path": str(input_path),
                    "count": len(imported_objects),
                    "objects": serialized,
                }
        except Exception as e:
            return False, str(e)

    def dataframe(
        self,
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
    ) -> Tuple[bool, Any]:
        """Preview or save Kosh dataframe output for a store, dataset, or ensemble.

        :param action: ``"preview"`` or ``"save"``.
        :param scope: ``"store"``, ``"dataset"``, or ``"ensemble"``.
        :param object_id: Dataset or ensemble ID or name when required by
            ``scope``.
        :param output_path: Output file or directory for ``action="save"``.
        :param output_format: ``"csv"`` or ``"json"`` for saved dataframes.
        :param data_columns: Optional dataframe column selection.
        :param data_criteria: Optional Kosh or Sina data filters.
        :param ensemble_tags: Optional ensemble tag filters for ensemble scope.
        :param include_ensemble_attributes: Include ensemble attributes for
            ensemble scope.
        :param include_ensemble_tags: Include ensemble tags for ensemble scope.
        :param overwrite: Replace an existing output file.
        :returns: Tuple of a success flag and a JSON-safe payload or error
            string.
        """
        try:
            normalized_action = require_action(action, {"preview", "save"})
            if not isinstance(scope, str):
                raise ValueError("scope must be 'store', 'dataset', or 'ensemble'.")
            normalized_scope = scope.strip().lower()
            if normalized_scope not in {"store", "dataset", "ensemble"}:
                raise ValueError(
                    f"Unknown scope '{scope}'. Must be 'store', 'dataset', or 'ensemble'."
                )
            with self.store_helper.connection() as store:
                df, resolved_id = self.store_helper.dataframe_for_scope(
                    store,
                    normalized_scope,
                    object_id,
                    data_columns,
                    data_criteria,
                    ensemble_tags,
                    include_ensemble_attributes,
                    include_ensemble_tags,
                )
                rows, columns = getattr(df, "shape", (None, None))
                try:
                    dataframe_columns = [str(column) for column in df.columns]
                except Exception:
                    dataframe_columns = []

                payload = {
                    "action": normalized_action,
                    "scope": normalized_scope,
                    "object_id": resolved_id,
                    "shape": [
                        int(rows) if rows is not None else None,
                        int(columns) if columns is not None else None,
                    ],
                    "columns": dataframe_columns,
                }

                if normalized_action == "preview":
                    payload["dataframe"] = self.serializer.serialize(df)
                    return True, payload

                normalized_format = require_action(
                    output_format,
                    {"csv", "json"},
                    param_name="output_format",
                )
                suffix = "json" if normalized_format == "json" else "csv"
                default_name = f"{normalized_scope}_dataframe.{suffix}"
                output = self._prepare_write_file(output_path, default_name, overwrite)

                if normalized_format == "csv":
                    df.to_csv(output, index=False)
                else:
                    df.to_json(output, orient="records")

                payload["output_format"] = normalized_format
                payload["path"] = str(output)
                return True, payload
        except Exception as e:
            return False, str(e)

    @staticmethod
    def _curve_values(values: Any, name: str) -> List[Any]:
        """Return curve values as a non-empty list.

        :param values: Raw curve values to validate.
        :param name: Parameter name used in validation errors.
        :returns: List of curve values.
        """
        if isinstance(values, (str, bytes)) or values is None:
            raise ValueError(f"{name} must be a non-empty list of values.")
        try:
            items = list(values)
        except TypeError as e:
            raise ValueError(f"{name} must be a non-empty list of values.") from e
        if not items:
            raise ValueError(f"{name} must be a non-empty list of values.")
        return items

    @staticmethod
    def _curve_name(value: Optional[str], fallback: Optional[str], param_name: str) -> str:
        """Return a valid Kosh curve name.

        :param value: Requested curve name.
        :param fallback: Default curve name when ``value`` is ``None``.
        :param param_name: Parameter name used in validation errors.
        :returns: Valid curve name.
        """
        if value is None:
            if fallback is None:
                raise ValueError(f"{param_name} must be a non-empty string.")
            return fallback
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{param_name} must be a non-empty string when provided.")
        return value.strip()

    def add_curve(
        self,
        dataset_id: str,
        name: str,
        x_values: List[float],
        y_values: List[float],
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
    ) -> Tuple[bool, Any]:
        """Add a 1D curve directly to a dataset.

        :param dataset_id: Dataset ID or name.
        :param name: Curve set name and resulting feature name.
        :param x_values: Independent curve values.
        :param y_values: Dependent curve values.
        :param x_label: Optional independent curve name.
        :param y_label: Optional dependent curve name.
        :returns: Tuple of a success flag and a JSON-safe payload or error
            string.
        """
        try:
            curve_set = self._curve_name(name, None, "name")
            x_curve = self._curve_name(x_label, "x", "x_label")
            y_curve = self._curve_name(y_label, "value", "y_label")
            if x_curve == y_curve:
                raise ValueError("x_label and y_label must resolve to different curve names.")

            x_data = self._curve_values(x_values, "x_values")
            y_data = self._curve_values(y_values, "y_values")
            if len(x_data) != len(y_data):
                raise ValueError("x_values and y_values must contain the same number of values.")

            with self.store_helper.connection() as store:
                dataset_id = self.store_helper.object_name_to_id(store, dataset_id, "dataset")
                dataset = store.open(dataset_id)

                x_path = dataset.add_curve(
                    x_data,
                    curve_set=curve_set,
                    curve_name=x_curve,
                    independent=True,
                )
                y_path = dataset.add_curve(
                    y_data,
                    curve_set=curve_set,
                    curve_name=y_curve,
                    independent=False,
                )

                return True, {
                    "dataset_id": dataset_id,
                    "feature_name": curve_set,
                    "x_curve": x_path,
                    "y_curve": y_path,
                    "points": len(x_data),
                }
        except Exception as e:
            return False, str(e)

    def delete(self, object_type: str, object_id: str) -> Tuple[bool, Any]:
        """Delete a dataset or ensemble record from the store.

        :param object_type: ``"dataset"`` or ``"ensemble"``.
        :param object_id: Object ID or name.
        :returns: Tuple of a success flag and a JSON-safe payload or error
            string.
        """
        try:
            with self.store_helper.connection() as store:
                obj_type = require_object_type(object_type)
                object_id = self.store_helper.object_name_to_id(store, object_id, obj_type)

                obj = store.open(object_id)
                if obj_type == "ensemble":
                    for dataset_id in list(obj.get_members(ids_only=True)):
                        obj.remove(dataset_id)
                else:
                    for ensemble_id in list(obj.get_ensembles(ids_only=True)):
                        ensemble = store.open(ensemble_id)
                        ensemble.remove(obj)

                store.delete(object_id)

                return True, {"id": object_id}
        except Exception as e:
            return False, str(e)
