"""JSON-safe response serializers for the Kosh MCP server."""

import math
from collections.abc import Iterable, Mapping
from itertools import islice
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from sina.model import Record

    HAS_SINA = True
except ImportError:
    HAS_SINA = False

# This module is larger than the simpler MCP servers because Kosh can return
# datasets, ensembles, sources, Sina records, arrays, and dataframes. Keep the
# response shaping here so tool methods can hand back bounded JSON summaries.


class KoshSerializer:
    """Serialize Kosh MCP responses into stable JSON-safe dictionaries."""

    CORE_ATTRIBUTE_NAMES = {"id", "name"}
    SOURCE_TOP_LEVEL_ATTRIBUTES = {"uri", "mime_type", "associated"}

    def __init__(self, max_items: int = 20, max_rows: int = 5, max_depth: int = 5):
        """Initialize serialization limits.

        :param max_items: Maximum number of items to keep from list-like values.
        :param max_rows: Maximum number of dataframe rows to include in previews.
        :param max_depth: Maximum recursive serialization depth before falling
            back to ``str`` output.
        """
        self.max_items = max_items
        self.max_rows = max_rows
        self.max_depth = max_depth

    def serialize_object(self, obj: Any, include_files: bool = True) -> Dict[str, Any]:
        """Serialize a Kosh dataset, ensemble, source, or Sina-backed object.

        :param obj: Kosh or Sina-backed object to serialize.
        :param include_files: Include associated file summaries when available.
        :returns: A JSON-safe object summary.
        """
        object_type = self._object_type(obj)
        if object_type == "dataset":
            return self.serialize_dataset(obj, include_files=include_files)
        if object_type == "ensemble":
            return self.serialize_ensemble(obj, include_files=include_files)
        if object_type == "source":
            return self.serialize_source(obj)
        if self._is_sina_record(obj):
            return self.serialize_record(obj)

        payload = {
            "object_type": object_type,
            "id": self._safe_attr(obj, "id"),
            "name": self._safe_attr(obj, "name"),
            "attributes": self._object_attributes(obj),
        }
        return self._drop_none(payload)

    def serialize_find_object(self, obj: Any, detail: str = "compact") -> Dict[str, Any]:
        """Serialize a Kosh object for bounded ``find`` responses.

        :param obj: Kosh object to serialize.
        :param detail: ``"compact"`` or ``"standard"`` summary level.
        :returns: A JSON-safe object summary for ``find`` responses.
        """
        normalized_detail = str(detail or "compact").strip().lower()
        if normalized_detail not in {"compact", "standard"}:
            raise ValueError("detail must be 'compact' or 'standard'.")

        object_type = self._object_type(obj)
        if object_type == "dataset":
            payload = {
                "object_type": "dataset",
                "id": self._safe_attr(obj, "id"),
                "name": self._safe_attr(obj, "name"),
                "attributes": self._object_attributes(obj),
            }
            if normalized_detail == "standard":
                curve_sets = self._curve_set_summary(obj)
                if curve_sets:
                    payload["curve_sets"] = curve_sets
            return self._drop_none(payload)
        if object_type == "ensemble":
            payload = {
                "object_type": "ensemble",
                "id": self._safe_attr(obj, "id"),
                "name": self._safe_attr(obj, "name"),
                "attributes": self._object_attributes(obj),
            }
            if normalized_detail == "standard":
                curve_sets = self._curve_set_summary(obj)
                if curve_sets:
                    payload["curve_sets"] = curve_sets
            return self._drop_none(payload)
        if object_type == "source":
            if normalized_detail == "standard":
                return self.serialize_source(obj)

            record = self._safe_record(obj)
            if record is not None:
                raw = self._record_raw(record)
                data = self._record_section(record, raw, "data", {})
                payload = {
                    "object_type": "source",
                    "id": self._record_id(record, raw),
                    "uri": self._record_data_value(data, "uri"),
                    "mime_type": self._record_data_value(data, "mime_type"),
                    "attributes": self._record_data_values(
                        data,
                        exclude=self.SOURCE_TOP_LEVEL_ATTRIBUTES,
                    ),
                }
                return self._drop_none(payload)

            return self._drop_none(
                {
                    "object_type": "source",
                    "id": self._safe_attr(obj, "id"),
                    "uri": self._safe_attr(obj, "uri"),
                    "mime_type": self._safe_attr(obj, "mime_type"),
                    "attributes": self._object_attributes(obj),
                }
            )
        return self.serialize_object(obj, include_files=False)

    def serialize_dataset(self, dataset: Any, include_files: bool = True) -> Dict[str, Any]:
        """Serialize a Kosh dataset without loading feature data.

        :param dataset: Dataset object to serialize.
        :param include_files: Include associated file summaries when available.
        :returns: A JSON-safe dataset summary.
        """
        payload = {
            "object_type": "dataset",
            "id": self._safe_attr(dataset, "id"),
            "name": self._safe_attr(dataset, "name"),
            "attributes": self._object_attributes(dataset),
            "ensembles": self._membership_summary(dataset, "get_ensembles", "ensemble_ids"),
        }

        if include_files:
            file_count, associated_files, files_truncated = self._associated_files(dataset)
            payload["associated_file_count"] = file_count
            payload["associated_files"] = associated_files
            payload["associated_files_truncated"] = files_truncated

        curve_sets = self._curve_set_summary(dataset)
        if curve_sets:
            payload["curve_sets"] = curve_sets

        return self._drop_none(payload)

    def serialize_ensemble(self, ensemble: Any, include_files: bool = True) -> Dict[str, Any]:
        """Serialize a Kosh ensemble with a bounded membership summary.

        :param ensemble: Ensemble object to serialize.
        :param include_files: Include associated file summaries when available.
        :returns: A JSON-safe ensemble summary.
        """
        payload = {
            "object_type": "ensemble",
            "id": self._safe_attr(ensemble, "id"),
            "name": self._safe_attr(ensemble, "name"),
            "attributes": self._object_attributes(ensemble),
            "members": self._membership_summary(ensemble, "get_members", "dataset_ids"),
        }

        if include_files:
            file_count, associated_files, files_truncated = self._associated_files(ensemble)
            payload["associated_file_count"] = file_count
            payload["associated_files"] = associated_files
            payload["associated_files_truncated"] = files_truncated

        curve_sets = self._curve_set_summary(ensemble)
        if curve_sets:
            payload["curve_sets"] = curve_sets

        return self._drop_none(payload)

    def serialize_source(
        self,
        source: Any,
        fallback_uri: Optional[str] = None,
        fallback_mime_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Serialize a Kosh source or file object.

        :param source: Source object or backing record to serialize.
        :param fallback_uri: URI to use when the object does not expose one.
        :param fallback_mime_type: MIME type to use when the object does not
            expose one.
        :returns: A JSON-safe source summary.
        """
        if self._is_sina_record(source):
            payload = self._source_from_record(
                source,
                fallback_uri=fallback_uri,
                fallback_mime_type=fallback_mime_type,
            )
        else:
            record = self._safe_record(source)
            if record is not None:
                payload = self._source_from_record(
                    record,
                    fallback_uri=fallback_uri,
                    fallback_mime_type=fallback_mime_type,
                )
            else:
                payload = {
                    "object_type": "source",
                    "id": self._safe_attr(source, "id"),
                    "uri": fallback_uri or self._safe_attr(source, "uri"),
                    "mime_type": fallback_mime_type or self._safe_attr(source, "mime_type"),
                    "attributes": self._object_attributes(source),
                }
        return self._drop_none(payload)

    def serialize_record(self, record: Any) -> Dict[str, Any]:
        """Serialize a Sina record into a bounded JSON-safe shape.

        :param record: Sina record or record-like mapping to serialize.
        :returns: A JSON-safe record summary.
        """
        raw = self._record_raw(record)
        data = self._record_section(record, raw, "data", {})
        files = self._record_section(record, raw, "files", {})

        payload = {
            "object_type": "record",
            "id": self._record_id(record, raw),
            "record_type": self._record_type(record, raw),
            "attributes": self._record_data_values(data),
        }

        if files:
            payload["files"] = [
                self._file_entry_summary(uri, file_info)
                for uri, file_info in files.items()
            ]

        curve_sets = self._record_section(record, raw, "curve_sets", {})
        if curve_sets:
            payload["curve_sets"] = self._curve_sets_from_mapping(curve_sets)

        return self._drop_none(payload)

    def serialize(self, value: Any, depth: int = 0) -> Any:
        """Convert feature values or tabular exports to JSON-safe data.

        :param value: Value to serialize.
        :param depth: Current recursive depth.
        :returns: A JSON-safe representation of ``value``.
        """
        if depth >= self.max_depth:
            return str(value)

        if value is None or self._is_json_scalar(value):
            return value

        if HAS_NUMPY and isinstance(value, np.generic):
            return self.serialize(value.item(), depth + 1)

        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")

        if self._is_sina_record(value):
            return self.serialize_record(value)

        if HAS_PANDAS and isinstance(value, pd.DataFrame):
            return self._serialize_dataframe(value)

        if HAS_PANDAS and isinstance(value, pd.Series):
            return self._serialize_series(value)

        if self._is_array_like(value):
            return self._serialize_array(value)

        if isinstance(value, Mapping):
            return {
                str(key): self.serialize(item, depth + 1)
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple, set)):
            items = list(value)
            if len(items) > self.max_items:
                return {
                    "kind": "sequence",
                    "item_count": len(items),
                    "truncated": True,
                    "items": [
                        self.serialize(item, depth + 1)
                        for item in items[: self.max_items]
                    ],
                }
            return [self.serialize(item, depth + 1) for item in items]

        if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
            items = list(islice(value, self.max_items + 1))
            truncated = len(items) > self.max_items
            serialized = [
                self.serialize(item, depth + 1)
                for item in items[: self.max_items]
            ]
            if truncated:
                return {
                    "kind": "iterable",
                    "truncated": True,
                    "items": serialized,
                }
            return serialized

        if hasattr(value, "to_dict") and callable(value.to_dict):
            try:
                return self.serialize(value.to_dict(), depth + 1)
            except Exception:
                pass

        if hasattr(value, "tolist") and callable(value.tolist):
            try:
                return self.serialize(value.tolist(), depth + 1)
            except Exception:
                pass

        return str(value)

    def _object_type(self, obj: Any) -> str:
        store = getattr(obj, "__store__", None)
        raw_type = getattr(obj, "__type__", None)

        if raw_type == getattr(store, "_dataset_record_type", None):
            return "dataset"
        if raw_type == getattr(store, "_ensembles_type", None):
            return "ensemble"
        if raw_type == getattr(store, "_sources_type", None):
            return "source"
        if raw_type:
            return str(raw_type)
        return type(obj).__name__

    def _object_attributes(self, obj: Any) -> Dict[str, Any]:
        if hasattr(obj, "list_attributes"):
            try:
                attributes = obj.list_attributes(dictionary=True)
                return {
                    str(key): self.serialize(value)
                    for key, value in attributes.items()
                    if key not in self.CORE_ATTRIBUTE_NAMES
                }
            except Exception:
                pass

        record = self._safe_record(obj)
        if record is not None:
            raw = self._record_raw(record)
            data = self._record_section(record, raw, "data", {})
            return self._record_data_values(data, exclude=self.CORE_ATTRIBUTE_NAMES)

        return {}

    def _associated_files(self, obj: Any) -> Tuple[int, List[Dict[str, Any]], bool]:
        record = self._safe_record(obj)
        if record is None:
            return 0, [], False

        raw = self._record_raw(record)
        files = self._record_section(record, raw, "files", {})
        if not isinstance(files, Mapping):
            return 0, [], False

        file_count = len(files)
        truncated = file_count > self.max_items
        summaries = []
        for uri, file_info in islice(files.items(), self.max_items):
            summary = self._file_entry_summary(uri, file_info)
            source_id = summary.get("id")
            source_record = self._safe_related_record(obj, source_id)
            if source_record is not None:
                summary = self._source_from_record(
                    source_record,
                    fallback_uri=str(uri),
                    fallback_mime_type=summary.get("mime_type"),
                    fallback_attributes=summary.get("attributes"),
                )
            summaries.append(summary)
        return file_count, summaries, truncated

    def _source_from_record(
        self,
        record: Any,
        fallback_uri: Optional[str] = None,
        fallback_mime_type: Optional[str] = None,
        fallback_attributes: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        raw = self._record_raw(record)
        data = self._record_section(record, raw, "data", {})
        values = self._record_data_values(data)
        attributes = self._record_data_values(
            data,
            exclude=self.SOURCE_TOP_LEVEL_ATTRIBUTES,
        )
        if fallback_attributes:
            attributes = {**fallback_attributes, **attributes}

        associated = self._record_data_value(data, "associated")
        if isinstance(associated, (list, tuple, set)):
            associated_dataset_ids = [str(item) for item in associated]
        elif associated is None:
            associated_dataset_ids = []
        else:
            associated_dataset_ids = [str(associated)]
        visible_dataset_ids = associated_dataset_ids[: self.max_items]
        dataset_ids_truncated = len(associated_dataset_ids) > self.max_items

        payload = {
            "object_type": "source",
            "id": self._record_id(record, raw),
            "uri": values.get("uri", fallback_uri),
            "mime_type": values.get("mime_type", fallback_mime_type),
            "associated_dataset_count": len(associated_dataset_ids),
            "associated_dataset_ids": visible_dataset_ids,
            "associated_dataset_ids_truncated": dataset_ids_truncated,
            "attributes": attributes,
        }
        return self._drop_none(payload)

    def _membership_summary(self, obj: Any, method_name: str, ids_key: str) -> Dict[str, Any]:
        method = getattr(obj, method_name, None)
        if not callable(method):
            return {"count": 0, ids_key: []}

        try:
            ids = [str(item) for item in method(ids_only=True)]
        except Exception:
            ids = []
        visible_ids = ids[: self.max_items]
        truncated = len(ids) > self.max_items

        return {
            "count": len(ids),
            ids_key: visible_ids,
            "truncated": truncated,
        }

    def _curve_set_summary(self, obj: Any) -> Optional[Dict[str, Any]]:
        record = self._safe_record(obj)
        if record is None:
            return None

        raw = self._record_raw(record)
        curve_sets = self._record_section(record, raw, "curve_sets", {})
        if not curve_sets:
            return None
        return self._curve_sets_from_mapping(curve_sets)

    def _curve_sets_from_mapping(self, curve_sets: Mapping) -> Dict[str, Any]:
        names = [str(name) for name in curve_sets.keys()]
        return {
            "count": len(names),
            "names": names[: self.max_items],
            "truncated": len(names) > self.max_items,
        }

    def _file_entry_summary(self, uri: Any, file_info: Any) -> Dict[str, Any]:
        info = file_info if isinstance(file_info, Mapping) else {}
        attributes = {
            str(key): self.serialize(value)
            for key, value in info.items()
            if key not in {"kosh_id", "mimetype", "mime_type"}
        }
        payload = {
            "object_type": "source",
            "id": info.get("kosh_id"),
            "uri": str(uri),
            "mime_type": info.get("mimetype", info.get("mime_type")),
            "attributes": attributes,
        }
        return self._drop_none(payload)

    def _record_data_values(
        self,
        data: Any,
        exclude: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        if not isinstance(data, Mapping):
            return {}

        excluded = exclude or set()
        values = {}
        for key, entry in data.items():
            if key in excluded:
                continue
            if isinstance(entry, Mapping) and "value" in entry:
                value = entry["value"]
            else:
                value = entry
            values[str(key)] = self.serialize(value)
        return values

    @staticmethod
    def _record_data_value(data: Any, key: str) -> Any:
        if not isinstance(data, Mapping) or key not in data:
            return None

        entry = data[key]
        if isinstance(entry, Mapping) and "value" in entry:
            return entry["value"]
        return entry

    def _safe_attr(self, obj: Any, name: str) -> Optional[Any]:
        try:
            value = getattr(obj, name)
        except Exception:
            return None
        return self.serialize(value)

    def _safe_record(self, obj: Any) -> Optional[Any]:
        get_record = getattr(obj, "get_record", None)
        if not callable(get_record):
            return None
        try:
            return get_record()
        except Exception:
            return None

    def _safe_related_record(self, obj: Any, record_id: Optional[str]) -> Optional[Any]:
        if not record_id:
            return None

        get_record = getattr(obj, "get_record", None)
        if not callable(get_record):
            return None
        try:
            return get_record(record_id)
        except Exception:
            return None

    def _is_sina_record(self, value: Any) -> bool:
        if HAS_SINA and isinstance(value, Record):
            return True

        if isinstance(value, Mapping) and {"id", "type", "data"}.issubset(value.keys()):
            return True

        raw = getattr(value, "raw", None)
        if isinstance(raw, Mapping) and {"id", "type", "data"}.issubset(raw.keys()):
            return True
        return False

    def _record_raw(self, record: Any) -> Dict[str, Any]:
        raw = getattr(record, "raw", None)
        if isinstance(raw, Mapping):
            return dict(raw)

        if isinstance(record, Mapping):
            return dict(record)

        to_dict = getattr(record, "to_dict", None)
        if callable(to_dict):
            try:
                raw = to_dict()
                if isinstance(raw, Mapping):
                    return dict(raw)
            except Exception:
                pass

        raw = getattr(record, "__dict__", {}).get("raw")
        if isinstance(raw, Mapping):
            return dict(raw)

        return {}

    def _record_section(self, record: Any, raw: Dict[str, Any], name: str, default: Any) -> Any:
        if name in raw:
            return raw[name]
        try:
            return record[name]
        except Exception:
            return default

    def _record_id(self, record: Any, raw: Dict[str, Any]) -> Optional[str]:
        record_id = raw.get("id")
        if record_id is None:
            record_id = getattr(record, "id", None)
        return str(record_id) if record_id is not None else None

    def _record_type(self, record: Any, raw: Dict[str, Any]) -> Optional[str]:
        record_type = raw.get("type")
        if record_type is None:
            record_type = getattr(record, "type", None)
        return str(record_type) if record_type is not None else None

    @staticmethod
    def _drop_none(payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: value
            for key, value in payload.items()
            if value is not None
        }

    def _is_json_scalar(self, value: Any) -> bool:
        if isinstance(value, (bool, int, float, str)):
            if isinstance(value, float):
                return not (math.isnan(value) or math.isinf(value))
            return True
        return False

    @staticmethod
    def _is_array_like(value: Any) -> bool:
        return (
            hasattr(value, "shape")
            and hasattr(value, "dtype")
            and not isinstance(value, (str, bytes))
        )

    def _serialize_array(self, value: Any) -> Dict[str, Any]:
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        size = getattr(value, "size", None)

        if shape is not None:
            try:
                shape = [int(dimension) for dimension in shape]
            except (TypeError, ValueError):
                shape = list(shape)

        payload = {
            "kind": "array",
            "shape": shape,
            "dtype": str(dtype) if dtype else None,
            "size": int(size) if size is not None else None,
        }

        if size is not None and size <= self.max_items:
            try:
                payload["data"] = self.serialize(value.tolist())
            except Exception:
                payload["data"] = str(value)
            return self._drop_none(payload)

        try:
            # Avoid copying the full array just to build a truncated preview.
            if hasattr(value, "flat"):
                preview = list(islice(value.flat, self.max_items))
            elif HAS_NUMPY:
                preview = [
                    item.item()
                    for item in islice(
                        np.nditer(value, flags=["refs_ok", "zerosize_ok"], order="C"),
                        self.max_items,
                    )
                ]
            else:
                raise TypeError("array preview is unavailable")
            payload["preview"] = self.serialize(preview)
            payload["truncated"] = True
        except Exception:
            payload["preview"] = str(value)

        return self._drop_none(payload)

    def _serialize_series(self, value: Any) -> Dict[str, Any]:
        length = len(value)
        dtype = getattr(value, "dtype", None)
        name = getattr(value, "name", None)
        sample = value.head(self.max_items) if length > self.max_items else value

        try:
            sample_values = sample.to_dict()
        except Exception:
            try:
                sample_values = sample.tolist()
            except Exception:
                sample_values = str(sample)

        return {
            "kind": "series",
            "name": name,
            "length": length,
            "dtype": str(dtype) if dtype else None,
            "truncated": length > self.max_items,
            "values": self.serialize(sample_values),
        }

    def _serialize_dataframe(self, value: Any) -> Dict[str, Any]:
        shape = getattr(value, "shape", (None, None))
        columns = getattr(value, "columns", [])

        try:
            columns = [str(column) for column in columns]
        except Exception:
            columns = []

        rows, cols = shape
        sample = value.head(self.max_rows) if rows and rows > self.max_rows else value

        try:
            records = sample.to_dict("records")
        except Exception:
            records = []

        return {
            "kind": "dataframe",
            "shape": [
                int(rows) if rows is not None else None,
                int(cols) if cols is not None else None,
            ],
            "columns": columns,
            "rows_shown": len(records),
            "truncated": bool(rows and rows > self.max_rows),
            "sample_rows": self.serialize(records),
        }
