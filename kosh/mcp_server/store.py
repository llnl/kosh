"""Kosh store, query, and name-resolution helpers for the MCP server."""

import logging
from contextlib import contextmanager
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple, Union
from urllib.parse import unquote, urlparse

try:
    from sqlalchemy.engine import make_url
except ImportError:  # pragma: no cover - Kosh normally brings SQLAlchemy.
    make_url = None

from .utils import (
    check_json_object,
    require_object_id,
    require_object_type,
)

try:
    import kosh
except ImportError as e:
    raise RuntimeError(f"Failed to import Kosh: {e}") from e

# Many MCP tools accept either IDs or user-facing names. Keep the Kosh query
# and resolution rules here so the operation methods can delegate to Kosh
# directly without reimplementing lookup logic in each tool.


class KoshStoreHelper:
    """Encapsulate Kosh store access and object resolution helpers."""

    def __init__(
        self,
        database_uri: str,
        connection_type: str,
        execution_options: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize Kosh store access settings.

        :param database_uri: URI or path for the Kosh store.
        :param connection_type: Kosh connection type.
        :param execution_options: Optional execution options passed to
            ``kosh.connect``.
        :param logger: Optional logger for connection cleanup diagnostics.
        """
        self.database_uri = database_uri
        self.connection_type = connection_type
        self.execution_options = check_json_object(execution_options, "execution_options")
        self.logger = logger or logging.getLogger("kosh_mcp_server")

    def configure(
        self,
        database_uri: str,
        connection_type: str,
        execution_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update runtime Kosh connection settings.

        :param database_uri: URI or path for the Kosh store.
        :param connection_type: Kosh connection type.
        :param execution_options: Optional execution options passed to
            ``kosh.connect``.
        """
        self.database_uri = database_uri
        self.connection_type = connection_type
        self.execution_options = check_json_object(execution_options, "execution_options")

    def validate_connection(self) -> None:
        """Open and close the configured Kosh store to validate connectivity."""
        with self.connection():
            pass

    @classmethod
    def _local_store_path(cls, database_uri: Any) -> Optional[Path]:
        """Return a local filesystem path for a path-backed store URI.

        :param database_uri: Store URI or filesystem path.
        :returns: ``Path`` for local file-backed targets, otherwise ``None``.
        """
        if not isinstance(database_uri, str):
            return None

        resolved_uri = database_uri.strip()
        if not resolved_uri or resolved_uri in {":memory:", "sqlite:///:memory:"}:
            return None

        if resolved_uri.startswith("file:"):
            parsed = urlparse(resolved_uri)
            if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
                return None
            path_text = parsed.path or resolved_uri[len("file:"):]
            if not path_text:
                return None
            return Path(unquote(path_text)).expanduser()

        if "://" not in resolved_uri:
            return Path(resolved_uri).expanduser()

        if make_url is None:
            return None

        try:
            parsed = make_url(resolved_uri)
        except Exception:
            return None

        driver_name = getattr(parsed, "drivername", "")
        database_path = getattr(parsed, "database", None)
        if (
            not driver_name.startswith("sqlite")
            or not database_path
            or database_path == ":memory:"
        ):
            return None
        return Path(unquote(database_path)).expanduser()

    @classmethod
    def describe_database_uri(cls, database_uri: Any) -> Dict[str, Any]:
        """Describe a store URI with an absolute local path when possible.

        :param database_uri: Store URI or filesystem path.
        :returns: Mapping describing the store target and resolved local path.
        """
        local_store_path = cls._local_store_path(database_uri)
        full_path = None
        store_exists = None
        if local_store_path is not None:
            resolved_path = local_store_path.resolve(strict=False)
            full_path = str(resolved_path)
            store_exists = resolved_path.exists()

        return {
            "database_uri": database_uri,
            "full_path": full_path,
            "store_exists": store_exists,
        }

    def describe_connection(self) -> Dict[str, Any]:
        """Describe the active Kosh store connection target.

        :returns: Mapping describing the configured active store.
        """
        return self.describe_database_uri(self.database_uri)

    @staticmethod
    def object_matches_type(obj: Any, object_type: Optional[str]) -> bool:
        """Check whether a Kosh object matches the requested canonical type.

        :param obj: Kosh object to inspect.
        :param object_type: Canonical object type to compare against.
        :returns: ``True`` when ``obj`` matches ``object_type``.
        """
        if object_type is None:
            return True

        store = getattr(obj, "__store__", None)
        obj_type = getattr(obj, "__type__", None)
        ensemble_type = getattr(store, "_ensembles_type", None)
        dataset_type = getattr(store, "_dataset_record_type", None)

        if object_type == "dataset":
            return obj_type == dataset_type
        if object_type == "ensemble":
            return obj_type == ensemble_type
        if object_type == "source":
            return obj_type == getattr(store, "_sources_type", None)
        return True

    @staticmethod
    def normalize_range_criteria(
        data_criteria: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Convert JSON range criteria to Sina ``DataRange`` objects.

        :param data_criteria: Raw JSON-style criteria mapping.
        :returns: Criteria mapping with range dictionaries converted to
            ``DataRange`` objects.
        """
        if data_criteria is None:
            return None
        if not isinstance(data_criteria, dict):
            raise ValueError("data_criteria must be a JSON object.")
        if not data_criteria:
            return data_criteria

        from sina.utils import DataRange

        normalized = {}
        for key, value in data_criteria.items():
            if isinstance(value, dict) and ("min" in value or "max" in value):
                normalized[key] = DataRange(
                    min=value.get("min"),
                    max=value.get("max"),
                    min_inclusive=value.get("min_inclusive", True),
                    max_inclusive=value.get("max_inclusive", False),
                )
            else:
                normalized[key] = value
        return normalized

    @staticmethod
    def _collect_limited(values: Iterable[Any], limit: Optional[int]) -> Tuple[List[Any], bool]:
        """Collect values up to ``limit`` while preserving truncation state.

        :param values: Iterable of values to collect.
        :param limit: Maximum number of values to keep, or ``None``.
        :returns: Tuple of collected items and whether more values existed.
        """
        if limit is None:
            return list(values), False

        items = list(islice(values, limit + 1))
        return items[:limit], len(items) > limit

    def _iter_store_graph(self, store: Any) -> Iterator[Any]:
        """Yield a store and its associated stores once in traversal order.

        :param store: Root Kosh store for traversal.
        :yields: Each reachable store exactly once.
        """
        visited: Set[Any] = set()

        def walk(current: Any) -> Iterator[Any]:
            identifier = getattr(current, "db_uri", None)
            if identifier is None:
                identifier = id(current)
            if identifier in visited:
                return
            visited.add(identifier)

            yield current
            for associated_store in getattr(current, "_associated_stores_", []):
                yield from walk(associated_store)

        yield from walk(store)

    def _fast_store_page(
        self,
        store: Any,
        object_type: str,
        data_criteria: Optional[Dict[str, Any]],
        ensemble_id: Optional[str],
        file_uri: Optional[str],
        page_size: int,
    ) -> Optional[Tuple[int, List[str], str]]:
        """Return a fast SQL count and first-page ID list for supported queries.

        :param store: Kosh store to query.
        :param object_type: Canonical object type being queried.
        :param data_criteria: Optional simple attribute filters.
        :param ensemble_id: Optional ensemble ID for dataset membership
            queries.
        :param file_uri: Optional associated file URI filter for dataset or
            ensemble queries.
        :param page_size: Maximum number of IDs to return for the first page.
        :returns: Tuple of exact count, first-page IDs, and count method for
            supported SQL paths, or ``None``.
        """
        try:
            sina_store = store.get_sina_store()
        except Exception:
            return None

        record_dao = getattr(sina_store, "_record_dao", None)
        session = getattr(record_dao, "session", None) if record_dao is not None else None
        if session is None:
            return None

        try:
            from sqlalchemy import func
            from sina.datastores import sql_schema
            from sina.utils import sort_and_standardize_criteria
        except Exception:
            return None

        normalized_criteria = self.normalize_range_criteria(data_criteria) or {}
        scalar_criteria = []
        string_criteria = []
        scalar_list_criteria = []
        string_list_criteria = []
        universal_criteria = []
        if normalized_criteria:
            try:
                (
                    scalar_criteria,
                    string_criteria,
                    scalar_list_criteria,
                    string_list_criteria,
                    universal_criteria,
                ) = sort_and_standardize_criteria(normalized_criteria)
            except ValueError:
                return None
            if scalar_list_criteria or string_list_criteria or universal_criteria:
                return None

        id_column = sql_schema.Record.id
        if object_type == "dataset":
            result_query = session.query(sql_schema.Record.id).filter(
                sql_schema.Record.type.notin_(store._kosh_reserved_record_types)
            )
            method = "sql_record_type"
            if ensemble_id is not None:
                result_query = result_query.join(
                    sql_schema.Relationship,
                    sql_schema.Record.id == sql_schema.Relationship.subject_id,
                ).filter(
                    sql_schema.Relationship.object_id == ensemble_id,
                    sql_schema.Relationship.predicate == store._ensemble_predicate,
                )
                method = "sql_ensemble_membership"
        elif object_type == "ensemble" and ensemble_id is None:
            result_query = session.query(sql_schema.Record.id).filter(
                sql_schema.Record.type == store._ensembles_type
            )
            method = "sql_record_type"
        elif (
            object_type == "source"
            and ensemble_id is None
            and not normalized_criteria
            and not file_uri
        ):
            result_query = session.query(sql_schema.Record.id).filter(
                sql_schema.Record.type == store._sources_type
            )
            method = "sql_record_type"
        else:
            return None

        if scalar_criteria:
            scalar_query = record_dao._generate_data_table_query(
                scalar_criteria,
                sql_schema.ScalarData,
            ).subquery()
            result_query = result_query.join(
                scalar_query,
                sql_schema.Record.id == scalar_query.c.id,
            )

        if string_criteria:
            string_query = record_dao._generate_data_table_query(
                string_criteria,
                sql_schema.StringData,
            ).subquery()
            result_query = result_query.join(
                string_query,
                sql_schema.Record.id == string_query.c.id,
            )

        if file_uri:
            if object_type not in {"dataset", "ensemble"}:
                return None

            document_query = session.query(sql_schema.Document.id)
            if "%" in file_uri:
                document_query = document_query.filter(sql_schema.Document.uri.like(file_uri))
            else:
                document_query = document_query.filter(sql_schema.Document.uri == file_uri)
            document_query = document_query.group_by(sql_schema.Document.id).subquery()
            result_query = result_query.join(
                document_query,
                sql_schema.Record.id == document_query.c.id,
            )

        if normalized_criteria or file_uri:
            method_parts = []
            if ensemble_id is not None:
                method_parts.append("ensemble_membership")
            if normalized_criteria:
                method_parts.append("data_criteria")
            if file_uri:
                method_parts.append("file_uri")
            method = "sql_" + "_".join(method_parts)

        result_query = result_query.distinct()
        total_count = int(
            session.query(func.count()).select_from(result_query.order_by(None).subquery()).scalar()
            or 0
        )
        ids: List[str] = []
        if page_size:
            ids = [
                str(row[0])
                for row in result_query.order_by(id_column).limit(page_size).all()
            ]

        return total_count, ids, method

    def fast_find_page(
        self,
        store: Any,
        object_type: str,
        ids: Optional[List[str]],
        data_criteria: Optional[Dict[str, Any]],
        ensemble_id: Optional[str],
        file_uri: Optional[str],
        owner_type: Optional[str],
        owner_id: Optional[str],
        page_size: int,
    ) -> Optional[Dict[str, Any]]:
        """Return a fast SQL-backed count and first-page handles when possible.

        :param store: Root Kosh store for the query.
        :param object_type: Canonical object type being queried.
        :param ids: Optional concrete object IDs.
        :param data_criteria: Optional attribute filter criteria.
        :param ensemble_id: Optional ensemble ID for dataset membership
            queries.
        :param file_uri: Optional associated file URI filter.
        :param owner_type: Optional owner type for source queries.
        :param owner_id: Optional owner ID for source queries.
        :param page_size: Maximum number of results to keep in the first page.
        :returns: Fast-path count and handle metadata, or ``None`` when no
            supported SQL path exists.
        """
        if ids or owner_type or owner_id:
            return None
        if ensemble_id and object_type != "dataset":
            return None

        total_count = 0
        handles: List[Tuple[Any, str]] = []
        remaining = page_size
        method: Optional[str] = None

        for current_store in self._iter_store_graph(store):
            page = self._fast_store_page(
                current_store,
                object_type=object_type,
                data_criteria=data_criteria,
                ensemble_id=ensemble_id,
                file_uri=file_uri,
                page_size=remaining,
            )
            if page is None:
                return None

            store_count, store_ids, store_method = page
            total_count += store_count
            handles.extend((current_store, object_id) for object_id in store_ids)
            remaining = max(0, remaining - len(store_ids))

            if method is None:
                method = store_method
            elif method != store_method:
                method = "mixed_sql_fast_path"

        return {
            "count": total_count,
            "handles": handles,
            "truncated": total_count > len(handles),
            "count_method": method or "sql_fast_path",
        }

    def object_name_to_id(self, store: Any, ref: str, object_type: str) -> str:
        """Resolve a user-facing object reference to a concrete Kosh object ID.

        :param store: Open Kosh store used for lookups.
        :param ref: User-facing object ID or name.
        :param object_type: Canonical object type for the lookup.
        :returns: Concrete Kosh object ID.
        """
        ref = require_object_id("object_ref", ref)
        canonical_type = require_object_type(object_type)

        try:
            obj = store.open(ref)
        except Exception:
            obj = None

        if obj is not None:
            if self.object_matches_type(obj, canonical_type):
                return obj.id
            raise ValueError(
                f"Reference '{ref}' resolved to a different object type than '{canonical_type}'."
            )

        if canonical_type == "ensemble":
            candidates = [str(item) for item in store.find_ensembles(name=ref, ids_only=True)]
        else:
            candidates = [
                str(item)
                for item in store.find(
                    name=ref,
                    types=store._dataset_record_type,
                    ids_only=True,
                )
            ]

        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise ValueError(f"{canonical_type.capitalize()} '{ref}' was not found.")

        raise ValueError(
            f"{canonical_type.capitalize()} name '{ref}' is ambiguous. Matching IDs: {candidates}"
        )

    def _iter_find_values(
        self,
        store: Any,
        object_type: str,
        ids: Optional[List[str]],
        data_criteria: Optional[Dict[str, Any]],
        ensemble_id: Optional[str],
        file_uri: Optional[str],
        owner_type: Optional[str],
        owner_id: Optional[str],
        ids_only: bool = False,
    ) -> Iterable[Any]:
        """Build the iterable used by Kosh-backed ``find`` fallbacks.

        :param store: Open Kosh store used for the query.
        :param object_type: Canonical object type being queried.
        :param ids: Optional concrete object IDs.
        :param data_criteria: Optional Kosh or Sina attribute filters.
        :param ensemble_id: Optional ensemble ID for dataset membership
            queries.
        :param file_uri: Optional associated file URI filter.
        :param owner_type: Optional owner type for source queries.
        :param owner_id: Optional owner ID for source queries.
        :param ids_only: Return IDs instead of loaded objects when ``True``.
        :returns: Iterable over matching IDs or objects.
        """
        criteria = self.normalize_range_criteria(data_criteria) or {}

        if object_type == "source":
            if ensemble_id:
                raise ValueError("ensemble_id filtering is not supported for source queries.")

            source_criteria = dict(criteria)
            if file_uri:
                source_criteria["uri"] = file_uri

            if owner_id or owner_type:
                if not owner_id or not owner_type:
                    raise ValueError(
                        "Source queries require both owner_type and owner_id when either is provided."
                    )
                canonical_owner_type = require_object_type(owner_type)
                resolved_owner_id = self.object_name_to_id(store, owner_id, canonical_owner_type)
                owner = store.open(resolved_owner_id)
                owner_kwargs = {"load_type": "source", **source_criteria}
                if ids_only:
                    owner_kwargs["ids_only"] = True

                sources = owner.find(**owner_kwargs)
                if ids:
                    wanted = set(ids)
                    if ids_only:
                        return (source_id for source_id in sources if source_id in wanted)
                    return (
                        source
                        for source in sources
                        if getattr(source, "id", None) in wanted
                    )
                return sources

            source_kwargs = {"types": store._sources_type, **source_criteria}
            if ids:
                source_kwargs["id_pool"] = ids
            if ids_only:
                source_kwargs["ids_only"] = True
            else:
                source_kwargs["load_type"] = "record"
            return store.find(**source_kwargs)

        if owner_id or owner_type:
            raise ValueError("owner_type and owner_id are only supported for source queries.")

        if ids:
            def direct_matches() -> Iterator[Any]:
                for obj_id in ids:
                    obj = store.open(obj_id)
                    if not self.object_matches_type(obj, object_type):
                        continue
                    yield obj_id if ids_only else obj

            return direct_matches()

        if ensemble_id:
            if object_type != "dataset":
                raise ValueError("ensemble_id filtering is only supported for dataset queries.")
            ensemble = store.open(ensemble_id)
            dataset_kwargs = dict(criteria)
            if file_uri:
                dataset_kwargs["file_uri"] = file_uri
            if ids_only:
                dataset_kwargs["ids_only"] = True
            return ensemble.find_datasets(**dataset_kwargs)

        query_kwargs = dict(criteria)
        if file_uri:
            query_kwargs["file_uri"] = file_uri
        if ids_only:
            query_kwargs["ids_only"] = True

        if object_type == "ensemble":
            return store.find_ensembles(**query_kwargs)
        return store.find(**query_kwargs)

    @contextmanager
    def connection(self):
        """Open a Kosh store for a single tool call and close it on exit.

        :yields: Open Kosh store connection.
        """
        uri = self.database_uri
        mode = self.connection_type
        if (
            isinstance(uri, str)
            and uri
            and not uri.startswith(":memory:")
            and "://" not in uri
        ):
            db_path = Path(uri).expanduser()
            if mode != "read":
                db_path.parent.mkdir(parents=True, exist_ok=True)
            uri = str(db_path)

        store = None
        try:
            store = kosh.connect(
                uri,
                connection_type=mode,
                execution_options=self.execution_options,
            )
            yield store
        finally:
            if store is not None:
                close_method = getattr(store, "close", None)
                if callable(close_method):
                    try:
                        close_method()
                    except Exception:
                        self.logger.debug("Failed to close Kosh store cleanly.", exc_info=True)

    def find_object_ids(
        self,
        store: Any,
        object_type: str,
        ids: Optional[List[str]],
        data_criteria: Optional[Dict[str, Any]],
        ensemble_id: Optional[str],
        file_uri: Optional[str],
        owner_type: Optional[str],
        owner_id: Optional[str],
        limit: Optional[int],
    ) -> Tuple[List[str], bool]:
        """Find matching object IDs using Kosh APIs.

        :param store: Open Kosh store used for the query.
        :param object_type: Canonical object type being queried.
        :param ids: Optional concrete object IDs.
        :param data_criteria: Optional Kosh or Sina attribute filters.
        :param ensemble_id: Optional ensemble ID for dataset membership
            queries.
        :param file_uri: Optional associated file URI filter.
        :param owner_type: Optional owner type for source queries.
        :param owner_id: Optional owner ID for source queries.
        :param limit: Maximum number of IDs to keep, or ``None``.
        :returns: Tuple of matching IDs and whether more matches existed.
        """
        values = self._iter_find_values(
            store,
            object_type=object_type,
            ids=ids,
            data_criteria=data_criteria,
            ensemble_id=ensemble_id,
            file_uri=file_uri,
            owner_type=owner_type,
            owner_id=owner_id,
            ids_only=True,
        )
        return self._collect_limited(values, limit)

    def count_objects(
        self,
        store: Any,
        object_type: str,
        ids: Optional[List[str]],
        data_criteria: Optional[Dict[str, Any]],
        ensemble_id: Optional[str],
        file_uri: Optional[str],
        owner_type: Optional[str],
        owner_id: Optional[str],
    ) -> int:
        """Count matches by iterating the Kosh-backed ID stream.

        :param store: Open Kosh store used for the query.
        :param object_type: Canonical object type being queried.
        :param ids: Optional concrete object IDs.
        :param data_criteria: Optional Kosh or Sina attribute filters.
        :param ensemble_id: Optional ensemble ID for dataset membership
            queries.
        :param file_uri: Optional associated file URI filter.
        :param owner_type: Optional owner type for source queries.
        :param owner_id: Optional owner ID for source queries.
        :returns: Exact count of matching objects.
        """
        values = self._iter_find_values(
            store,
            object_type=object_type,
            ids=ids,
            data_criteria=data_criteria,
            ensemble_id=ensemble_id,
            file_uri=file_uri,
            owner_type=owner_type,
            owner_id=owner_id,
            ids_only=True,
        )
        return sum(1 for _ in values)

    def find_objects(
        self,
        store: Any,
        object_type: str,
        ids: Optional[List[str]],
        data_criteria: Optional[Dict[str, Any]],
        ensemble_id: Optional[str],
        file_uri: Optional[str],
        owner_type: Optional[str],
        owner_id: Optional[str],
        limit: Optional[int],
    ) -> Tuple[List[Any], bool]:
        """Find Kosh objects using explicit Kosh APIs.

        :param store: Open Kosh store used for the query.
        :param object_type: Canonical object type to query.
        :param ids: Optional concrete object IDs to resolve directly.
        :param data_criteria: Optional Kosh or Sina attribute filters.
        :param ensemble_id: Optional ensemble ID for dataset membership queries.
        :param file_uri: Optional associated file URI filter.
        :param owner_type: Optional owner type for source queries.
        :param owner_id: Optional owner ID for source queries.
        :param limit: Maximum number of results to keep, or ``None``.
        :returns: Tuple of matching objects and whether the results were
            truncated.
        """
        values = self._iter_find_values(
            store,
            object_type=object_type,
            ids=ids,
            data_criteria=data_criteria,
            ensemble_id=ensemble_id,
            file_uri=file_uri,
            owner_type=owner_type,
            owner_id=owner_id,
            ids_only=False,
        )
        return self._collect_limited(values, limit)

    @staticmethod
    def _normalize_data_columns(
        data_columns: Optional[Union[str, List[str]]],
    ) -> Union[str, List[str]]:
        """Normalize dataframe column selection for Kosh dataframe calls.

        :param data_columns: Column name or names requested by the caller.
        :returns: Column selection in the form expected by Kosh dataframe APIs.
        """
        if data_columns is None:
            return []
        if isinstance(data_columns, str):
            if not data_columns.strip():
                raise ValueError("data_columns must not contain empty column names.")
            return data_columns.strip()

        try:
            columns = list(data_columns)
        except TypeError as e:
            raise ValueError("data_columns must be a string or list of strings.") from e
        if any(not isinstance(column, str) or not column.strip() for column in columns):
            raise ValueError("data_columns must not contain empty column names.")
        return [column.strip() for column in columns]

    def dataframe_for_scope(
        self,
        store: Any,
        scope: str,
        object_id: Optional[str],
        data_columns: Optional[Union[str, List[str]]],
        data_criteria: Optional[Dict[str, Any]],
        ensemble_tags: Optional[Dict[str, Any]],
        include_ensemble_attributes: bool,
        include_ensemble_tags: bool,
    ) -> Tuple[Any, Optional[str]]:
        """Build a dataframe using the matching Kosh dataframe API.

        :param store: Open Kosh store used for dataframe generation.
        :param scope: ``"store"``, ``"dataset"``, or ``"ensemble"``.
        :param object_id: Dataset or ensemble ID or name when required by
            ``scope``.
        :param data_columns: Optional column selection.
        :param data_criteria: Optional Kosh or Sina data filters.
        :param ensemble_tags: Optional ensemble tag filters.
        :param include_ensemble_attributes: Include ensemble attributes for
            ensemble dataframes.
        :param include_ensemble_tags: Include ensemble tags for ensemble
            dataframes.
        :returns: Tuple of the constructed dataframe and the resolved object ID,
            if one applies to the requested scope.
        """
        columns = self._normalize_data_columns(data_columns)
        criteria = self.normalize_range_criteria(data_criteria) or {}
        kwargs = {"data": criteria} if criteria else {}

        if scope == "store":
            if object_id:
                raise ValueError("object_id is only valid for dataset and ensemble dataframe scopes.")
            return store.to_dataframe(data_columns=columns, **kwargs), None

        if scope == "dataset":
            resolved_id = self.object_name_to_id(store, object_id, "dataset")
            dataset = store.open(resolved_id)
            return dataset.to_dataframe(data_columns=columns, **kwargs), resolved_id

        resolved_id = self.object_name_to_id(store, object_id, "ensemble")
        ensemble = store.open(resolved_id)
        normalized_tags = self.normalize_range_criteria(ensemble_tags) or {}
        if normalized_tags:
            kwargs["ensemble_tags"] = normalized_tags
        return (
            ensemble.to_dataframe(
                data_columns=columns,
                include_ensemble_attributes=include_ensemble_attributes,
                include_ensemble_tags=include_ensemble_tags,
                **kwargs,
            ),
            resolved_id,
        )
