"""Transform registry and execution-graph helpers for the Kosh MCP server."""

import importlib
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .utils import check_json_object, require_object_id

# Kosh transforms/operators have a distinct registry and execution-graph flow,
# so they live apart from the CRUD-style operations rather than adding more
# branching to the main tool module.

DEFAULT_TRANSFORMER_REGISTRY = {
    "DBSCAN": ("kosh.transformers", "DBSCAN"),
    "Delta": ("kosh.transformers.npy", "Delta"),
    "KMeans": ("kosh.transformers", "KMeans"),
    "KoshSimpleNpCache": ("kosh.transformers", "KoshSimpleNpCache"),
    "Shuffle": ("kosh.transformers.npy", "Shuffle"),
    "SidreFeatureMetrics": ("kosh.transformers", "SidreFeatureMetrics"),
    "Splitter": ("kosh.transformers", "Splitter"),
    "StandardScaler": ("kosh.transformers", "StandardScaler"),
    "Take": ("kosh.transformers.npy", "Take"),
}

DEFAULT_OPERATOR_REGISTRY = {
    "KoshLNorm": ("kosh.operators", "KoshLNorm"),
}


class KoshTransformHelper:
    """Encapsulate transform registry lookup and graph construction."""

    def __init__(
        self,
        serializer: Any,
        resolve_object_id: Callable[[Any, str, str], str],
        transformer_registry: Optional[Dict[str, Tuple[str, str]]] = None,
        operator_registry: Optional[Dict[str, Tuple[str, str]]] = None,
    ):
        """Initialize transform helpers and registries.

        :param serializer: Response serializer used for catalog output.
        :param resolve_object_id: Callable that resolves user-facing object
            references to concrete dataset IDs.
        :param transformer_registry: Optional transformer registry override.
        :param operator_registry: Optional operator registry override.
        """
        self.serializer = serializer
        self.resolve_object_id = resolve_object_id
        self.transformer_registry = (
            transformer_registry
            if transformer_registry is not None
            else DEFAULT_TRANSFORMER_REGISTRY
        )
        self.operator_registry = (
            operator_registry
            if operator_registry is not None
            else DEFAULT_OPERATOR_REGISTRY
        )

    def configure(
        self,
        serializer: Any,
        transformer_registry: Dict[str, Tuple[str, str]],
        operator_registry: Dict[str, Tuple[str, str]],
    ) -> None:
        """Update serializer and registries used for transform operations.

        :param serializer: Response serializer used for catalog output.
        :param transformer_registry: Transformer registry to expose.
        :param operator_registry: Operator registry to expose.
        """
        self.serializer = serializer
        self.transformer_registry = transformer_registry
        self.operator_registry = operator_registry

    @staticmethod
    def _registry_class(
        name: Any,
        registry: Dict[str, Tuple[str, str]],
        kind: str,
    ) -> Tuple[str, Any]:
        """Resolve an allow-listed transformer or operator class by name.

        :param name: Requested transformer or operator name.
        :param registry: Registry mapping names to module and class paths.
        :param kind: Registry kind used in validation errors.
        :returns: Tuple of canonical registry name and resolved class object.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{kind} spec requires a non-empty name.")

        requested = name.strip()
        canonical = requested if requested in registry else None
        if canonical is None:
            matches = [
                candidate
                for candidate in registry
                if candidate.lower() == requested.lower()
            ]
            canonical = matches[0] if len(matches) == 1 else None
        if canonical is None:
            allowed = ", ".join(sorted(registry))
            raise ValueError(f"Unknown {kind} '{name}'. Supported {kind}s: {allowed}.")

        module_name, class_name = registry[canonical]
        module = importlib.import_module(module_name)
        return canonical, getattr(module, class_name)

    def catalog_entries(
        self,
        registry: Dict[str, Tuple[str, str]],
        kind: str,
    ) -> Dict[str, Dict[str, Any]]:
        """Return registered transform classes with their Kosh type mappings.

        :param registry: Registry mapping names to module and class paths.
        :param kind: Registry kind used in validation errors.
        :returns: JSON-safe catalog of registered classes.
        """
        entries = {}
        for name in sorted(registry):
            _, cls = self._registry_class(name, registry, kind)
            try:
                signature = str(inspect.signature(cls))
            except (TypeError, ValueError):
                signature = None
            entries[name] = {
                key: value
                for key, value in {
                    "signature": signature,
                    "types": self.serializer.serialize(getattr(cls, "types", {})),
                }.items()
                if value is not None
            }
        return entries

    def normalize_transform_spec(
        self,
        spec: Any,
        registry: Dict[str, Tuple[str, str]],
        kind: str,
    ) -> Tuple[str, Any, Dict[str, Any]]:
        """Return a transform class and constructor options from a JSON spec.

        :param spec: Transformer or operator specification.
        :param registry: Registry mapping names to module and class paths.
        :param kind: Registry kind used in validation errors.
        :returns: Tuple of canonical name, class object, and constructor
            options.
        """
        if isinstance(spec, str):
            name = spec
            options = {}
        elif isinstance(spec, dict):
            name = spec.get("name", spec.get("type"))
            options = check_json_object(spec.get("options"), f"{kind} options")
        else:
            raise ValueError(f"{kind} spec must be a string or JSON object.")

        resolved_name, cls = self._registry_class(name, registry, kind)
        return resolved_name, cls, options

    def instantiate_transformers(
        self,
        specs: Optional[Union[str, Dict[str, Any], List[Any]]],
    ) -> Tuple[List[Any], List[Dict[str, Any]]]:
        """Instantiate allow-listed Kosh transformers from JSON specs.

        :param specs: Transformer specification or specifications.
        :returns: Tuple of instantiated transformers and JSON-safe summaries.
        """
        if specs is None:
            raw_specs = []
        elif isinstance(specs, (str, dict)):
            raw_specs = [specs]
        else:
            try:
                raw_specs = list(specs)
            except TypeError as e:
                raise ValueError("transformers must be a string, JSON object, or list.") from e

        transformers = []
        summaries = []
        for spec in raw_specs:
            name, cls, options = self.normalize_transform_spec(
                spec,
                self.transformer_registry,
                "transformer",
            )
            transformers.append(cls(**options))
            summaries.append({"name": name, "options": options})
        return transformers, summaries

    @staticmethod
    def normalize_transform_inputs(
        inputs: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]],
    ) -> List[Dict[str, Any]]:
        """Normalize one or more transform input specs.

        :param inputs: Input specification or specifications for a transform
            graph.
        :returns: List of JSON input specifications.
        """
        if inputs is None:
            raise ValueError("transform action='apply' requires at least one input.")
        if isinstance(inputs, dict):
            normalized = [inputs]
        else:
            try:
                normalized = list(inputs)
            except TypeError as e:
                raise ValueError("inputs must be a JSON object or list of JSON objects.") from e
        if not normalized:
            raise ValueError("transform action='apply' requires at least one input.")
        return normalized

    def transform_input_graph(
        self,
        store: Any,
        input_spec: Dict[str, Any],
    ) -> Tuple[Any, Dict[str, Any]]:
        """Build a Kosh execution graph from one transform input spec.

        :param store: Open Kosh store used to resolve datasets.
        :param input_spec: JSON specification for one transform input.
        :returns: Tuple of the execution graph and a JSON-safe input summary.
        """
        if not isinstance(input_spec, dict):
            raise ValueError("Each transform input must be a JSON object.")

        dataset_ref = require_object_id("dataset_id", input_spec.get("dataset_id"))
        dataset_id = self.resolve_object_id(store, dataset_ref, "dataset")
        dataset = store.open(dataset_id)
        feature_name = input_spec.get("feature_name")
        if not isinstance(feature_name, str) or not feature_name.strip():
            raise ValueError("transform input feature_name must be a non-empty string.")
        feature_name = feature_name.strip()
        source_id = input_spec.get("source_id")
        resolved_source_id = (
            require_object_id("source_id", source_id)
            if source_id
            else None
        )
        use_cache = input_spec.get("use_cache", True)
        if not isinstance(use_cache, bool):
            raise ValueError("transform input use_cache must be a boolean.")
        input_options = check_json_object(input_spec.get("options"), "input options")
        transformers, transformer_summary = self.instantiate_transformers(
            input_spec.get("transformers")
        )

        graph = dataset.get_execution_graph(
            feature=feature_name,
            Id=resolved_source_id,
            transformers=transformers,
            use_cache=use_cache,
            **input_options,
        )
        return graph, {
            "dataset_id": dataset_id,
            "feature_name": feature_name,
            "source_id": resolved_source_id,
            "use_cache": use_cache,
            "transformers": transformer_summary,
        }
