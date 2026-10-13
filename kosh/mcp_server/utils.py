"""Reusable helpers for Kosh MCP server modules."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

# Keep only small cross-module helpers here. Kosh-specific store, transform,
# and serialization behavior should stay in their dedicated modules.


def resolve_runtime_value(value: Any, fallback: str) -> str:
    """Return a normalized runtime string value.

    :param value: Runtime value from config or environment.
    :param fallback: Value to use when ``value`` is missing or empty.
    :returns: A non-empty string value.
    """
    if value is None:
        return fallback
    if not isinstance(value, str):
        return str(value)

    stripped = value.strip()
    return stripped or fallback


def get_env_var(
    var_name: str,
    default: Optional[str] = None,
    required: bool = False,
) -> Optional[str]:
    """Return an environment variable using the current mada-tools helper.

    Falls back to a local implementation with the same contract when the
    optional ``mada-tools`` dependency is unavailable. This keeps utility-level
    imports and tests working even though actual MCP server startup still
    requires ``mada-tools``.

    :param var_name: Environment variable name to read.
    :param default: Fallback value when the variable is unset.
    :param required: Raise when the variable must be present.
    :returns: The resolved environment value or ``default``.
    """
    mada_get_env_var = _load_mada_get_env_var()
    if mada_get_env_var is None:
        return _fallback_get_env_var(
            var_name,
            default=default,
            required=required,
        )
    return mada_get_env_var(var_name, default=default, required=required)


def _load_mada_get_env_var() -> Optional[Callable[..., Optional[str]]]:
    """Return the mada-tools environment helper when available.

    :returns: ``mada_tools.shared.env.get_env_var`` or ``None`` when
        ``mada-tools`` is not installed.
    """
    try:
        from mada_tools.shared.env import get_env_var as mada_get_env_var
    except ModuleNotFoundError as e:
        if not (e.name or "").startswith("mada_tools"):
            raise
        return None
    return mada_get_env_var


def _fallback_get_env_var(
    var_name: str,
    default: Optional[str] = None,
    required: bool = False,
) -> Optional[str]:
    """Local compatibility shim for ``mada_tools.shared.env.get_env_var``.

    :param var_name: Environment variable name to read.
    :param default: Fallback value when the variable is unset.
    :param required: Raise when the variable must be present.
    :returns: The resolved environment value or ``default``.
    """
    value = os.getenv(var_name, default)
    if required and value is None:
        raise ValueError(f"Required environment variable {var_name} is not set")
    return value


def environment_value(name: str, fallback: str) -> str:
    """Return a normalized environment value.

    :param name: Environment variable name to read.
    :param fallback: Value to use when the variable is unset or empty.
    :returns: A non-empty environment value.
    """
    value = get_env_var(name)
    if value is not None and value.strip():
        return value.strip()
    return fallback


def execution_options_from_environment(var_name: str) -> Dict[str, Any]:
    """Parse execution options for ``kosh.connect`` from the environment.

    :param var_name: Environment variable containing a JSON object.
    :returns: Parsed execution options, or an empty dictionary.
    """
    value = get_env_var(var_name)
    if value is None or not value.strip():
        return {}

    try:
        execution_options = json.loads(value)
    except json.JSONDecodeError as e:
        raise ValueError(f"{var_name} must be a JSON object.") from e

    if not isinstance(execution_options, dict):
        raise ValueError(f"{var_name} must be a JSON object.")
    return execution_options


def normalize_store_aliases(
    store_aliases: Optional[Dict[str, Any]],
    param_name: str = "store_aliases",
) -> Dict[str, str]:
    """Validate and normalize a mapping of store aliases to database URIs.

    :param store_aliases: Alias mapping to validate.
    :param param_name: Parameter name used in validation errors.
    :returns: Normalized alias mapping.
    """
    if store_aliases is None:
        return {}

    if not isinstance(store_aliases, dict):
        raise ValueError(
            f"{param_name} must be a JSON object mapping aliases to database URIs."
        )

    normalized = {}
    for alias, database_uri in store_aliases.items():
        if not isinstance(alias, str) or not alias.strip():
            raise ValueError(f"{param_name} aliases must be non-empty strings.")
        if not isinstance(database_uri, str) or not database_uri.strip():
            raise ValueError(
                f"{param_name} values must be non-empty database URI strings."
            )

        alias_name = alias.strip()
        if alias_name in normalized:
            raise ValueError(f"{param_name} contains duplicate alias '{alias_name}'.")
        normalized[alias_name] = database_uri.strip()

    return normalized


def set_loglevel(logger: logging.Logger, loglevel: str) -> None:
    """Apply a configured logging level to the process and server logger.

    :param logger: Server logger to update.
    :param loglevel: Logging level name such as ``"INFO"`` or ``"WARNING"``.
    """
    log_level = getattr(logging, loglevel.upper(), logging.WARNING)
    logging.getLogger().setLevel(log_level)
    logger.setLevel(log_level)


def check_json_object(
    value: Optional[Dict[str, Any]],
    param_name: str = "value",
) -> Dict[str, Any]:
    """Validate an optional JSON object value.

    :param value: Value expected to be a dictionary or ``None``.
    :param param_name: Parameter name used in validation errors.
    :returns: ``value`` or an empty dictionary when ``value`` is ``None``.
    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{param_name} must be a JSON object.")
    return value


def check_path(path: Optional[str], param_name: str = "path") -> Path:
    """Validate a non-empty filesystem path.

    :param path: Path value to validate.
    :param param_name: Parameter name used in validation errors.
    :returns: Expanded filesystem path.
    """
    if not isinstance(path, str) or not path.strip():
        raise ValueError(f"Missing or empty '{param_name}'. Provide a valid path.")
    return Path(path).expanduser()


def require_object_id(param_name: str, value: Optional[str]) -> str:
    """Validate that an object identifier is present and non-empty.

    :param param_name: Parameter name used in validation errors.
    :param value: Object identifier to validate.
    :returns: The original identifier value.
    """
    if not value or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"Missing or empty '{param_name}'. Provide a valid object ID.")
    return value


def require_object_type(object_type: str, allow_source: bool = False) -> str:
    """Normalize and validate an MCP object type.

    :param object_type: User-provided object type.
    :param allow_source: Allow ``"source"`` in addition to dataset and ensemble.
    :returns: Canonical lower-case object type.
    """
    allowed = {"dataset", "ensemble"}
    if allow_source:
        allowed.add("source")
    allowed_text = ", ".join(f"'{item}'" for item in sorted(allowed))

    if not isinstance(object_type, str):
        raise ValueError(f"object_type must be one of: {allowed_text}.")
    normalized = object_type.strip().lower()
    if normalized not in allowed:
        raise ValueError(
            f"Unknown object_type '{object_type}'. Must be one of: {allowed_text}."
        )
    return normalized


def require_action(
    action: str,
    allowed: Iterable[str],
    param_name: str = "action",
) -> str:
    """Normalize and validate an action name.

    :param action: User-provided action string.
    :param allowed: Allowed action names.
    :param param_name: Parameter name used in validation errors.
    :returns: Canonical lower-case action value.
    """
    if not isinstance(action, str):
        raise ValueError(f"{param_name} must be a string.")
    normalized = action.strip().lower()
    allowed_set = set(allowed)
    if normalized not in allowed_set:
        allowed_text = ", ".join(f"'{item}'" for item in sorted(allowed_set))
        raise ValueError(
            f"Unknown {param_name} '{action}'. Must be one of: {allowed_text}."
        )
    return normalized


def coerce_string_list(
    values: Optional[Union[str, List[str]]],
    param_name: str,
    required: bool = False,
) -> List[str]:
    """Normalize a string or list of strings.

    :param values: String value or iterable of string values.
    :param param_name: Parameter name used in validation errors.
    :param required: Require at least one value.
    :returns: List of stripped string values.
    """
    if values is None:
        if required:
            raise ValueError(f"Missing or empty '{param_name}'.")
        return []
    if isinstance(values, str):
        items = [values]
    else:
        try:
            items = list(values)
        except TypeError as e:
            raise ValueError(
                f"{param_name} must be a string or list of strings."
            ) from e
    if required and not items:
        raise ValueError(f"Missing or empty '{param_name}'.")
    if any(not isinstance(item, str) or not item.strip() for item in items):
        raise ValueError(f"{param_name} entries must be non-empty strings.")
    return [item.strip() for item in items]


def normalize_limit(limit: Optional[int]) -> Optional[int]:
    """Normalize a result limit.

    :param limit: Maximum number of results to keep.
    :returns: Non-negative integer limit, or ``None`` for unbounded queries.
    """
    if limit is None:
        return None
    try:
        normalized = int(limit)
    except (TypeError, ValueError) as e:
        raise ValueError("limit must be a non-negative integer.") from e
    if normalized < 0:
        raise ValueError("limit must be a non-negative integer.")
    return normalized or None


def as_list(value: Any) -> List[Any]:
    """Return a value as a list without splitting strings or dictionaries.

    :param value: Value to normalize.
    :returns: List representation of ``value``.
    """
    if value is None:
        return []
    if isinstance(value, (str, bytes, dict)):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def safe_filename(value: str) -> str:
    """Return a conservative filename component for generated outputs.

    :param value: Raw filename component.
    :returns: Sanitized filename component safe for generated paths.
    """
    safe = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in value
    )
    return safe or "object"


def coerce_file_uris(file_uris: Union[str, List[str]]) -> List[str]:
    """Normalize one or more file URIs.

    :param file_uris: File URI string or iterable of file URI strings.
    :returns: List of non-empty file URI strings.
    """
    if isinstance(file_uris, str):
        uris = [file_uris]
    else:
        try:
            uris = list(file_uris or [])
        except TypeError as e:
            raise ValueError(
                "associate_or_dissociate_file requires a file URI string "
                "or list of file URI strings."
            ) from e
    if not uris or any(not isinstance(uri, str) or not uri.strip() for uri in uris):
        raise ValueError(
            "associate_or_dissociate_file requires one or more non-empty file URIs."
        )
    return uris
