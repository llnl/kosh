"""Focused tests for the Kosh MCP server and operation adapter."""

import asyncio
import json
import logging
import sys
import types
from pathlib import Path

import kosh
import kosh.mcp_server.utils as mcp_utils
import pytest

from kosh.mcp_server.operations import KoshMCPOperations
from kosh.mcp_server.server import _KoshServerMixin
from kosh.mcp_server.store import KoshStoreHelper
from kosh.mcp_server.utils import (
    execution_options_from_environment,
)


def _payload(result):
    success, payload = result
    assert success, payload
    return payload


def _failure(result):
    success, payload = result
    assert not success
    return payload


def _ops(tmp_path, name="kosh_mcp.sqlite", **kwargs):
    return KoshMCPOperations(database_uri=str(tmp_path / name), **kwargs)


class RecordingMCP:
    def __init__(self):
        self.tools = {}

    def tool(self):
        def decorator(func):
            self.tools[func.__name__] = func
            return func

        return decorator


class FakeBaseMCPServer:
    def __init__(self, server_name, description=None):
        self.server_name = server_name
        self.description = description
        self.mcp = None
        self.last_run_tool = None

    async def run_tool(self, func, *args, background=True, **kwargs):
        self.last_run_tool = {
            "func": func,
            "args": args,
            "kwargs": dict(kwargs),
            "background": background,
        }
        success, payload = func(*args, **kwargs)
        assert success, payload
        return payload


class FakeOperations:
    def __init__(self):
        self.configured = None
        self.validated = False
        self.calls = []
        self.find_kwargs = None
        self.database_uri = None
        self.store_aliases = {}

    def configure(
        self,
        database_uri=None,
        connection_type=None,
        execution_options=None,
        store_aliases=None,
    ):
        self.configured = {
            "database_uri": database_uri,
            "connection_type": connection_type,
            "execution_options": dict(execution_options or {}),
            "store_aliases": dict(store_aliases or {}),
        }
        self.database_uri = database_uri
        self.store_aliases = dict(store_aliases or {})

    def validate_connection(self):
        self.validated = True

    def _record_call(self, method_name, *args, **kwargs):
        call = {
            "name": method_name,
            "args": args,
            "kwargs": dict(kwargs),
        }
        self.calls.append(call)
        return True, {
            "called": method_name,
            "args": list(args),
            "kwargs": dict(kwargs),
        }

    def find(self, **kwargs):
        self.find_kwargs = dict(kwargs)
        return self._record_call("find", **kwargs)

    def store_connection(self, **kwargs):
        action = kwargs.get("action", "current")
        alias = kwargs.get("alias")
        database_uri = kwargs.get("database_uri")
        if action == "connect" and alias:
            self.database_uri = self.store_aliases.get(alias, f"alias:{alias}")
        elif action == "connect" and database_uri:
            self.database_uri = database_uri
        return self._record_call("store_connection", **kwargs)

    def create_dataset_or_ensemble(self, *args, **kwargs):
        return self._record_call("create_dataset_or_ensemble", *args, **kwargs)

    def associate_or_dissociate_file(self, *args, **kwargs):
        return self._record_call("associate_or_dissociate_file", *args, **kwargs)

    def update(self, *args, **kwargs):
        return self._record_call("update", *args, **kwargs)

    def load(self, *args, **kwargs):
        return self._record_call("load", *args, **kwargs)

    def transform(self, *args, **kwargs):
        return self._record_call("transform", *args, **kwargs)

    def import_or_export(self, *args, **kwargs):
        return self._record_call("import_or_export", *args, **kwargs)

    def dataframe(self, *args, **kwargs):
        return self._record_call("dataframe", *args, **kwargs)

    def add_curve(self, *args, **kwargs):
        return self._record_call("add_curve", *args, **kwargs)

    def delete(self, *args, **kwargs):
        return self._record_call("delete", *args, **kwargs)


class KoshServerForTest(_KoshServerMixin, FakeBaseMCPServer):
    pass


class ScaleTransformer(kosh.transformers.KoshTransformer):
    types = {"numpy": ["numpy"]}

    def __init__(self, factor=1):
        super().__init__(factor=factor)
        self.factor = factor

    def transform(self, input_, format):
        return input_ * self.factor


class AddOperator(kosh.operators.KoshOperator):
    types = {"numpy": ["numpy"]}

    def operate(self, *inputs, **kwargs):
        result = inputs[0]
        for value in inputs[1:]:
            result = result + value
        return result


def _register_test_transformer(monkeypatch):
    module = types.ModuleType("kosh_mcp_test_transformers")
    module.ScaleTransformer = ScaleTransformer
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(
        KoshMCPOperations,
        "TRANSFORMER_REGISTRY",
        {"Scale": (module.__name__, "ScaleTransformer")},
    )
    monkeypatch.setattr(KoshMCPOperations, "OPERATOR_REGISTRY", {})


def _register_test_operator(monkeypatch):
    module = types.ModuleType("kosh_mcp_test_operators")
    module.AddOperator = AddOperator
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(
        KoshMCPOperations,
        "OPERATOR_REGISTRY",
        {"Add": (module.__name__, "AddOperator")},
    )


def test_kosh_mcp_server_registers_tool_surface_and_uses_environment(monkeypatch):
    monkeypatch.setenv("KOSH_DATABASE_URI", "configured.sqlite")
    monkeypatch.setenv("KOSH_CONNECTION_TYPE", "read")
    monkeypatch.setenv("KOSH_EXECUTION_OPTIONS", '{"timeout": 30}')
    monkeypatch.setenv(
        "KOSH_STORE_ALIASES_JSON",
        '{"scratch":"scratch.sqlite","archive":"archive.sqlite"}',
    )
    monkeypatch.setenv("KOSH_LOGLEVEL", "INFO")

    server = KoshServerForTest()
    server.operations = FakeOperations()
    server.mcp = RecordingMCP()

    server._register_tools()

    assert set(server.mcp.tools) == {
        "store_connection",
        "find",
        "create_dataset_or_ensemble",
        "associate_or_dissociate_file",
        "update",
        "load",
        "transform",
        "import_or_export",
        "dataframe",
        "add_curve",
        "delete",
    }
    assert server.operations.configured == {
        "database_uri": "configured.sqlite",
        "connection_type": "read",
        "execution_options": {"timeout": 30},
        "store_aliases": {
            "scratch": "scratch.sqlite",
            "archive": "archive.sqlite",
        },
    }
    assert server.operations.validated

    result = asyncio.run(
        server.mcp.tools["find"](
            "dataset",
            ids="run_001",
            mode="ids",
            page_size=5,
            detail="standard",
            allow_expensive=True,
        )
    )

    assert result["called"] == "find"
    assert server.last_run_tool["background"] is False
    assert server.operations.find_kwargs == {
        "object_type": "dataset",
        "ids": "run_001",
        "data_criteria": None,
        "ensemble_id": None,
        "file_uri": None,
        "owner_type": None,
        "owner_id": None,
        "mode": "ids",
        "page_size": 5,
        "detail": "standard",
        "allow_expensive": True,
    }


def test_get_env_var_prefers_current_mada_tools_helper(monkeypatch):
    calls = {}

    def fake_get_env_var(var_name, default=None, required=False):
        calls["var_name"] = var_name
        calls["default"] = default
        calls["required"] = required
        return "from-mada-tools"

    monkeypatch.setattr(
        mcp_utils,
        "_load_mada_get_env_var",
        lambda: fake_get_env_var,
    )

    assert (
        mcp_utils.get_env_var(
            "KOSH_DATABASE_URI",
            default="fallback.sqlite",
            required=True,
        )
        == "from-mada-tools"
    )
    assert calls == {
        "var_name": "KOSH_DATABASE_URI",
        "default": "fallback.sqlite",
        "required": True,
    }


def test_get_env_var_falls_back_without_mada_tools(monkeypatch):
    monkeypatch.setattr(mcp_utils, "_load_mada_get_env_var", lambda: None)

    monkeypatch.setenv("KOSH_DATABASE_URI", "configured.sqlite")
    assert mcp_utils.get_env_var("KOSH_DATABASE_URI") == "configured.sqlite"

    monkeypatch.delenv("KOSH_DATABASE_URI", raising=False)
    assert (
        mcp_utils.get_env_var(
            "KOSH_DATABASE_URI",
            default="fallback.sqlite",
        )
        == "fallback.sqlite"
    )
    with pytest.raises(
        ValueError,
        match="Required environment variable KOSH_DATABASE_URI is not set",
    ):
        mcp_utils.get_env_var("KOSH_DATABASE_URI", required=True)


def test_kosh_mcp_server_forwards_all_tool_arguments(monkeypatch):
    for name in (
        "KOSH_DATABASE_URI",
        "KOSH_CONNECTION_TYPE",
        "KOSH_EXECUTION_OPTIONS",
        "KOSH_STORE_ALIASES_JSON",
        "KOSH_LOGLEVEL",
    ):
        monkeypatch.delenv(name, raising=False)

    server = KoshServerForTest()
    server.operations = FakeOperations()
    server.mcp = RecordingMCP()
    server._register_tools()

    cases = [
        {
            "tool": "store_connection",
            "args": (),
            "kwargs": {},
            "expected_args": (),
            "expected_kwargs": {
                "action": "current",
                "alias": None,
                "database_uri": None,
            },
            "expected_server_database_uri": KoshMCPOperations.DEFAULT_DATABASE_URI,
        },
        {
            "tool": "store_connection",
            "args": (),
            "kwargs": {
                "action": "connect",
                "alias": "scratch",
            },
            "expected_args": (),
            "expected_kwargs": {
                "action": "connect",
                "alias": "scratch",
                "database_uri": None,
            },
            "expected_server_database_uri": "alias:scratch",
        },
        {
            "tool": "find",
            "args": ("dataset",),
            "kwargs": {
                "ids": ["run_001", "run_002"],
                "data_criteria": {"shot": {"min": 10}},
                "ensemble_id": "campaign",
                "file_uri": "input.json",
                "owner_type": "dataset",
                "owner_id": "run_001",
                "mode": "summary",
                "page_size": 7,
                "detail": "standard",
                "allow_expensive": True,
            },
            "expected_args": (),
            "expected_kwargs": {
                "object_type": "dataset",
                "ids": ["run_001", "run_002"],
                "data_criteria": {"shot": {"min": 10}},
                "ensemble_id": "campaign",
                "file_uri": "input.json",
                "owner_type": "dataset",
                "owner_id": "run_001",
                "mode": "summary",
                "page_size": 7,
                "detail": "standard",
                "allow_expensive": True,
            },
        },
        {
            "tool": "create_dataset_or_ensemble",
            "args": ("dataset",),
            "kwargs": {
                "name": "run_001",
                "metadata": {"shot": 12},
            },
            "expected_args": ("dataset",),
            "expected_kwargs": {
                "name": "run_001",
                "metadata": {"shot": 12},
            },
        },
        {
            "tool": "associate_or_dissociate_file",
            "args": ("dataset", "run_001", ["a.dat", "b.dat"]),
            "kwargs": {
                "mime_types": ["text/plain", "text/plain"],
                "metadata": [{"label": "a"}, {"label": "b"}],
                "absolute_path": False,
                "operation": "associate",
                "loader_kwargs": {"delimiter": ","},
                "long_sha": True,
                "preload_features": True,
            },
            "expected_args": ("dataset", "run_001", ["a.dat", "b.dat"]),
            "expected_kwargs": {
                "mime_types": ["text/plain", "text/plain"],
                "metadata": [{"label": "a"}, {"label": "b"}],
                "absolute_path": False,
                "operation": "associate",
                "loader_kwargs": {"delimiter": ","},
                "long_sha": True,
                "preload_features": True,
            },
        },
        {
            "tool": "update",
            "args": ("dataset", "run_001"),
            "kwargs": {
                "attributes": {"status": "ready"},
                "add_to_ensemble": "campaign",
                "remove_from_ensemble": "campaign",
                "inherit_attributes": False,
                "ensemble_tags": {"fold": "train"},
            },
            "expected_args": ("dataset", "run_001"),
            "expected_kwargs": {
                "attributes": {"status": "ready"},
                "add_to_ensemble": "campaign",
                "remove_from_ensemble": "campaign",
                "inherit_attributes": False,
                "ensemble_tags": {"fold": "train"},
            },
        },
        {
            "tool": "load",
            "args": ("run_001",),
            "kwargs": {
                "feature_name": ["A", "B"],
                "action": "get",
                "source_id": "source-1",
                "output_format": "dict",
                "group": True,
                "use_cache": False,
                "verbose": True,
                "options": {"filters": "all"},
                "list_features": True,
            },
            "expected_args": ("run_001",),
            "expected_kwargs": {
                "feature_name": ["A", "B"],
                "action": "get",
                "source_id": "source-1",
                "output_format": "dict",
                "group": True,
                "use_cache": False,
                "verbose": True,
                "options": {"filters": "all"},
                "list_features": True,
            },
        },
        {
            "tool": "transform",
            "args": (),
            "kwargs": {
                "action": "apply",
                "inputs": [
                    {"dataset_id": "run_001", "feature_name": "signal"},
                    {"dataset_id": "run_002", "feature_name": "signal"},
                ],
                "operator": {"name": "Add", "options": {"scale": 2}},
                "output_format": "numpy",
                "options": {"cache": False},
            },
            "expected_args": (),
            "expected_kwargs": {
                "action": "apply",
                "inputs": [
                    {"dataset_id": "run_001", "feature_name": "signal"},
                    {"dataset_id": "run_002", "feature_name": "signal"},
                ],
                "operator": {"name": "Add", "options": {"scale": 2}},
                "output_format": "numpy",
                "options": {"cache": False},
            },
        },
        {
            "tool": "import_or_export",
            "args": ("export", "/tmp/out.json"),
            "kwargs": {
                "object_type": "ensemble",
                "object_ids": ["campaign"],
                "match_attributes": ["name"],
                "merge_handler": "preserve",
                "overwrite": True,
            },
            "expected_args": ("export", "/tmp/out.json"),
            "expected_kwargs": {
                "object_type": "ensemble",
                "object_ids": ["campaign"],
                "match_attributes": ["name"],
                "merge_handler": "preserve",
                "overwrite": True,
            },
        },
        {
            "tool": "dataframe",
            "args": ("save", "dataset"),
            "kwargs": {
                "object_id": "run_001",
                "output_path": "/tmp/run_001.csv",
                "output_format": "csv",
                "data_columns": ["shot", "status"],
                "data_criteria": {"shot": {"min": 10}},
                "ensemble_tags": {"fold": "train"},
                "include_ensemble_attributes": False,
                "include_ensemble_tags": False,
                "overwrite": True,
            },
            "expected_args": ("save", "dataset"),
            "expected_kwargs": {
                "object_id": "run_001",
                "output_path": "/tmp/run_001.csv",
                "output_format": "csv",
                "data_columns": ["shot", "status"],
                "data_criteria": {"shot": {"min": 10}},
                "ensemble_tags": {"fold": "train"},
                "include_ensemble_attributes": False,
                "include_ensemble_tags": False,
                "overwrite": True,
            },
        },
        {
            "tool": "add_curve",
            "args": ("run_001", "signal", [0.0, 1.0], [10.0, 11.0]),
            "kwargs": {
                "x_label": "time",
                "y_label": "temperature",
            },
            "expected_args": ("run_001", "signal", [0.0, 1.0], [10.0, 11.0]),
            "expected_kwargs": {
                "x_label": "time",
                "y_label": "temperature",
            },
        },
        {
            "tool": "delete",
            "args": ("ensemble", "campaign"),
            "kwargs": {},
            "expected_args": ("ensemble", "campaign"),
            "expected_kwargs": {},
        },
    ]

    for case in cases:
        result = asyncio.run(
            server.mcp.tools[case["tool"]](*case["args"], **case["kwargs"])
        )

        assert result["called"] == case["tool"]
        assert server.last_run_tool["background"] is False
        assert server.operations.calls[-1] == {
            "name": case["tool"],
            "args": case["expected_args"],
            "kwargs": case["expected_kwargs"],
        }
        if "expected_server_database_uri" in case:
            assert server.database_uri == case["expected_server_database_uri"]


def test_kosh_mcp_server_environment_falls_back_to_defaults_and_validates_json(
    monkeypatch,
):
    monkeypatch.setenv("KOSH_DATABASE_URI", "   ")
    monkeypatch.setenv("KOSH_CONNECTION_TYPE", "")
    monkeypatch.setenv("KOSH_EXECUTION_OPTIONS", "  ")
    monkeypatch.setenv("KOSH_STORE_ALIASES_JSON", "  ")
    monkeypatch.setenv("KOSH_LOGLEVEL", "not-a-real-level")

    server = KoshServerForTest()
    server.operations = FakeOperations()
    server._configure_operations_from_environment()

    assert server.database_uri == server.DEFAULT_DATABASE_URI
    assert server.connection_type == server.DEFAULT_CONNECTION_TYPE
    assert server.execution_options == {}
    assert server.store_aliases == {}
    assert server.operations.configured == {
        "database_uri": server.DEFAULT_DATABASE_URI,
        "connection_type": server.DEFAULT_CONNECTION_TYPE,
        "execution_options": {},
        "store_aliases": {},
    }
    assert server.logger.level == logging.WARNING

    monkeypatch.setenv("KOSH_EXECUTION_OPTIONS", "[]")
    with pytest.raises(
        ValueError,
        match="KOSH_EXECUTION_OPTIONS must be a JSON object.",
    ):
        execution_options_from_environment(server.ENV_EXECUTION_OPTIONS)

    monkeypatch.setenv("KOSH_EXECUTION_OPTIONS", "{")
    with pytest.raises(
        ValueError,
        match="KOSH_EXECUTION_OPTIONS must be a JSON object.",
    ):
        execution_options_from_environment(server.ENV_EXECUTION_OPTIONS)


def test_kosh_mcp_server_store_alias_environment_parses_and_validates_json(
    monkeypatch,
):
    server = KoshServerForTest()
    server.operations = FakeOperations()

    monkeypatch.setenv(
        "KOSH_STORE_ALIASES_JSON",
        '{"scratch":" scratch.sqlite ","archive":"archive.sqlite"}',
    )
    server._configure_operations_from_environment()

    assert server.store_aliases == {
        "scratch": "scratch.sqlite",
        "archive": "archive.sqlite",
    }
    assert server.operations.configured["store_aliases"] == {
        "scratch": "scratch.sqlite",
        "archive": "archive.sqlite",
    }

    monkeypatch.setenv("KOSH_STORE_ALIASES_JSON", "[]")
    with pytest.raises(
        ValueError,
        match="KOSH_STORE_ALIASES_JSON must be a JSON object mapping aliases to database URIs.",
    ):
        server._configure_operations_from_environment()

    monkeypatch.setenv("KOSH_STORE_ALIASES_JSON", '{"":"store.sqlite"}')
    with pytest.raises(
        ValueError,
        match="KOSH_STORE_ALIASES_JSON aliases must be non-empty strings.",
    ):
        server._configure_operations_from_environment()

    monkeypatch.setenv("KOSH_STORE_ALIASES_JSON", '{"scratch":""}')
    with pytest.raises(
        ValueError,
        match="KOSH_STORE_ALIASES_JSON values must be non-empty database URI strings.",
    ):
        server._configure_operations_from_environment()


def test_kosh_mcp_operations_store_connection_switches_by_alias_and_warns_on_new_store(
    tmp_path,
    monkeypatch,
):
    scratch_path = tmp_path / "scratch.sqlite"
    ops = _ops(
        tmp_path,
        store_aliases={"scratch": str(scratch_path)},
    )
    monkeypatch.setattr(
        ops,
        "validate_connection",
        lambda: scratch_path.touch(),
    )

    connected = _payload(
        ops.store_connection(action="connect", alias="scratch")
    )

    assert connected["status"] == "connected"
    assert connected["action"] == "connect"
    assert connected["selection_type"] == "alias"
    assert connected["alias"] == "scratch"
    assert connected["database_uri"] == str(scratch_path)
    assert connected["full_path"] == str(scratch_path.resolve())
    assert connected["connection_type"] == "write"
    assert connected["store_exists"] is True
    assert connected["store_existed_before_connect"] is False
    assert connected["created_new_store"] is True
    assert connected["warnings"] == [
        f"Local store path '{scratch_path}' did not exist and was created."
    ]
    assert ops.database_uri == str(scratch_path)
    assert scratch_path.is_file()


def test_kosh_mcp_operations_store_connection_switches_by_database_uri(
    tmp_path,
    monkeypatch,
):
    explicit_path = tmp_path / "explicit.sqlite"
    explicit_path.touch()

    ops = _ops(tmp_path)
    monkeypatch.setattr(ops, "validate_connection", lambda: None)
    connected = _payload(
        ops.store_connection(action="connect", database_uri=str(explicit_path))
    )

    assert connected["status"] == "connected"
    assert connected["action"] == "connect"
    assert connected["selection_type"] == "database_uri"
    assert connected["alias"] is None
    assert connected["database_uri"] == str(explicit_path)
    assert connected["full_path"] == str(explicit_path.resolve())
    assert connected["store_exists"] is True
    assert connected["store_existed_before_connect"] is True
    assert connected["created_new_store"] is False
    assert connected["warnings"] == []
    assert ops.database_uri == str(explicit_path)


def test_kosh_mcp_operations_store_connection_reports_current_store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ops = KoshMCPOperations(database_uri="relative.sqlite")

    connected = _payload(ops.store_connection(action="current"))

    expected_path = (tmp_path / "relative.sqlite").resolve()
    assert connected["status"] == "connected"
    assert connected["action"] == "current"
    assert connected["selection_type"] == "current"
    assert connected["alias"] is None
    assert connected["database_uri"] == "relative.sqlite"
    assert connected["full_path"] == str(expected_path)
    assert connected["connection_type"] == "write"
    assert connected["store_exists"] is False
    assert connected["store_existed_before_connect"] is None
    assert connected["created_new_store"] is False
    assert connected["warnings"] == []


def test_kosh_mcp_operations_store_connection_rejects_multiple_selectors(tmp_path):
    ops = _ops(tmp_path)

    assert (
        "store_connection(action='connect') requires exactly one of alias or database_uri."
        in _failure(
            ops.store_connection(
                action="connect",
                alias="scratch",
                database_uri="scratch.sqlite",
            )
        )
    )


def test_kosh_mcp_operations_store_connection_rejects_current_with_selector(tmp_path):
    ops = _ops(tmp_path)

    assert (
        "store_connection(action='current') does not accept alias or database_uri."
        in _failure(ops.store_connection(action="current", alias="scratch"))
    )


def test_kosh_mcp_operations_store_connection_rejects_unknown_alias(tmp_path):
    ops = _ops(tmp_path, store_aliases={"scratch": "scratch.sqlite"})

    failure = _failure(ops.store_connection(action="connect", alias="missing"))

    assert "Unknown store alias 'missing'." in failure
    assert "Available store aliases: scratch." in failure


def test_kosh_mcp_operations_store_connection_restores_previous_store_after_failure(
    tmp_path,
    monkeypatch,
):
    ops = _ops(tmp_path)
    original_uri = ops.database_uri

    def _raise_validation_error():
        raise RuntimeError("boom")

    monkeypatch.setattr(ops, "validate_connection", _raise_validation_error)

    failure = _failure(
        ops.store_connection(
            action="connect",
            database_uri=str(tmp_path / "broken.sqlite"),
        )
    )

    assert "boom" in failure
    assert ops.database_uri == original_uri


def test_kosh_mcp_operations_lists_transform_catalog_without_store(
    tmp_path,
    monkeypatch,
):
    _register_test_transformer(monkeypatch)
    ops = _ops(tmp_path)

    catalog = _payload(ops.transform(action="list"))

    assert catalog["action"] == "list"
    assert catalog["operators"] == {}
    assert "Scale" in catalog["transformers"]
    assert catalog["transformers"]["Scale"]["types"] == {"numpy": ["numpy"]}


def test_kosh_store_helper_resolves_names_with_filtered_kosh_queries():
    helper = KoshStoreHelper("store.sqlite", "write")

    class FakeStore:
        _dataset_record_type = "dataset"

        def __init__(self):
            self.find_calls = []
            self.find_ensembles_calls = []

        def open(self, ref):
            raise KeyError(ref)

        def find(self, **kwargs):
            self.find_calls.append(kwargs)
            return iter(["dataset-1"])

        def find_ensembles(self, **kwargs):
            self.find_ensembles_calls.append(kwargs)
            return iter(["ensemble-1"])

    store = FakeStore()

    assert helper.object_name_to_id(store, "run_001", "dataset") == "dataset-1"
    assert store.find_calls == [
        {
            "name": "run_001",
            "types": "dataset",
            "ids_only": True,
        }
    ]
    assert helper.object_name_to_id(store, "campaign", "ensemble") == "ensemble-1"
    assert store.find_ensembles_calls == [{"name": "campaign", "ids_only": True}]


def test_kosh_store_helper_reports_ambiguous_name_matches():
    helper = KoshStoreHelper("store.sqlite", "write")

    class FakeStore:
        _dataset_record_type = "dataset"

        def open(self, ref):
            raise KeyError(ref)

        def find(self, **kwargs):
            return iter(["dataset-1", "dataset-2"])

    with pytest.raises(ValueError, match="Dataset name 'run_001' is ambiguous."):
        helper.object_name_to_id(FakeStore(), "run_001", "dataset")


def test_kosh_mcp_operations_cover_primary_real_store_workflow(tmp_path, monkeypatch):
    _register_test_transformer(monkeypatch)
    ops = _ops(tmp_path)
    ops.validate_connection()

    dataset = _payload(
        ops.create_dataset_or_ensemble(
            "dataset",
            name="run_001",
            metadata={"shot": 12, "status": "new"},
        )
    )
    ensemble = _payload(
        ops.create_dataset_or_ensemble(
            "ensemble",
            name="campaign",
            metadata={"stage": "screening"},
        )
    )
    dataset_id = dataset["id"]
    ensemble_id = ensemble["id"]
    assert "associated_file_count" not in dataset
    assert "associated_files" not in dataset
    assert "associated_file_count" not in ensemble
    assert "associated_files" not in ensemble

    associated = _payload(
        ops.associate_or_dissociate_file(
            "dataset",
            "run_001",
            ["a.dat", "b.dat"],
            mime_types=["text/plain", "text/plain"],
            metadata={"source": "simulation"},
            absolute_path=False,
            loader_kwargs={"delimiter": ","},
        )
    )
    assert associated["owner_id"] == dataset_id
    assert associated["file_count"] == 2
    assert associated["loader_kwargs_applied"] is True

    sources = _payload(
        ops.find(
            "source",
            owner_type="dataset",
            owner_id="run_001",
            file_uri="a.dat",
        )
    )
    assert sources["mode"] == "objects"
    assert sources["returned"] == 1
    assert sources["results"][0]["uri"] == "a.dat"

    updated = _payload(
        ops.update(
            "dataset",
            "run_001",
            attributes={"status": "ready"},
            add_to_ensemble="campaign",
            inherit_attributes=False,
            ensemble_tags={"fold": "train"},
        )
    )
    assert updated["operations"] == [
        "updated 1 attribute(s)",
        f"added to ensemble '{ensemble_id}'",
    ]
    assert "associated_file_count" not in updated["object"]
    assert "associated_files" not in updated["object"]

    found = _payload(
        ops.find(
            "dataset",
            data_criteria={"shot": {"min": 10, "max": 20}},
            ensemble_id="campaign",
        )
    )
    assert found["mode"] == "objects"
    assert found["returned"] == 1
    assert found["results"][0]["id"] == dataset_id

    curve = _payload(
        ops.add_curve(
            "run_001",
            "signal",
            x_values=[0.0, 1.0, 2.0],
            y_values=[10.0, 11.0, 12.0],
            x_label="time",
            y_label="temperature",
        )
    )
    assert curve["feature_name"] == "signal"
    assert curve["points"] == 3

    features = _payload(ops.load("run_001", action="list_features"))
    assert {"signal", "signal/time", "signal/temperature"}.issubset(
        set(features["features"])
    )

    loaded = _payload(
        ops.load("run_001", feature_name="signal/temperature", output_format="numpy")
    )
    assert loaded["data"]["data"] == [10.0, 11.0, 12.0]

    transformed = _payload(
        ops.transform(
            inputs={
                "dataset_id": "run_001",
                "feature_name": "signal/temperature",
                "transformers": {"name": "Scale", "options": {"factor": 2}},
            },
            output_format="numpy",
        )
    )
    assert transformed["inputs"][0]["transformers"] == [
        {"name": "Scale", "options": {"factor": 2}}
    ]
    assert transformed["data"]["data"] == [20.0, 22.0, 24.0]

    dataframe = _payload(
        ops.dataframe(
            "save",
            "ensemble",
            object_id="campaign",
            output_path=str(tmp_path / "campaign.json"),
            output_format="json",
        )
    )
    assert dataframe["shape"][0] == 1
    assert "shot" in dataframe["columns"]
    assert Path(dataframe["path"]).is_file()

    exported = _payload(
        ops.import_or_export(
            "export",
            str(tmp_path / "run_001_export.json"),
            object_type="dataset",
            object_ids="run_001",
        )
    )
    assert exported["object_ids"] == [dataset_id]
    assert Path(exported["paths"][0]).is_file()

    deleted = _payload(ops.delete("dataset", "run_001"))
    assert deleted["id"] == dataset_id


def test_kosh_mcp_operations_cover_source_updates_and_dissociate(tmp_path):
    ops = _ops(tmp_path)
    pgm_path = (
        Path(__file__).resolve().parent / "baselines" / "images" / "buffalo.pgm"
    )

    dataset = _payload(
        ops.create_dataset_or_ensemble(
            "dataset",
            name="run_001",
            metadata={"shot": 12},
        )
    )

    associated = _payload(
        ops.associate_or_dissociate_file(
            "dataset",
            "run_001",
            "a.dat",
            mime_types="text/plain",
            metadata={"source": "simulation"},
            absolute_path=False,
        )
    )
    source_id = associated["source_ids"][0]

    updated_source = _payload(
        ops.update("source", source_id, attributes={"label": "raw"})
    )
    assert updated_source["operations"] == ["updated 1 attribute(s)"]
    assert updated_source["object"]["id"] == source_id
    assert updated_source["object"]["attributes"]["label"] == "raw"

    _payload(
        ops.associate_or_dissociate_file(
            "dataset",
            "run_001",
            str(pgm_path),
            mime_types="pgm",
        )
    )
    described = _payload(
        ops.load(
            "run_001",
            feature_name="image",
            action="describe_feature",
        )
    )
    assert described["action"] == "describe_feature"
    assert described["feature_name"] == "image"
    assert described["description"]["format"].startswith("pgm")
    assert described["description"]["max_value"] == 255

    dissociated = _payload(
        ops.associate_or_dissociate_file(
            "dataset",
            "run_001",
            "a.dat",
            absolute_path=False,
            operation="remove",
        )
    )
    assert dissociated["owner_id"] == dataset["id"]
    assert dissociated["operation"] == "dissociate"
    assert dissociated["file_count"] == 1
    assert dissociated["files"] == ["a.dat"]

    remaining_sources = _payload(
        ops.find(
            "source",
            owner_type="dataset",
            owner_id="run_001",
            file_uri="a.dat",
        )
    )
    assert remaining_sources["returned"] == 0


def test_kosh_mcp_operations_cover_preview_export_truncation_and_delete_ensemble(
    tmp_path,
):
    ops = _ops(tmp_path)

    first_dataset = _payload(
        ops.create_dataset_or_ensemble(
            "dataset",
            name="run_001",
            metadata={"shot": 12, "status": "new"},
        )
    )
    second_dataset = _payload(
        ops.create_dataset_or_ensemble(
            "dataset",
            name="run_002",
            metadata={"shot": 13, "status": "ready"},
        )
    )
    ensemble = _payload(
        ops.create_dataset_or_ensemble(
            "ensemble",
            name="campaign",
            metadata={"stage": "screening"},
        )
    )

    added = _payload(
        ops.update("dataset", "run_001", add_to_ensemble="campaign")
    )
    assert added["operations"] == [f"added to ensemble '{ensemble['id']}'"]

    limited = _payload(ops.find("dataset", page_size=1))
    assert limited["mode"] == "summary"
    assert limited["count"] == 2
    assert limited["returned"] == 1
    assert limited["truncated"] is True
    assert "ensembles" not in limited["results"][0]

    preview = _payload(
        ops.dataframe(
            "preview",
            "store",
            data_columns=["shot", "status"],
        )
    )
    assert preview["action"] == "preview"
    assert preview["scope"] == "store"
    assert preview["object_id"] is None
    assert preview["dataframe"]["kind"] == "dataframe"
    assert "shot" in preview["columns"]

    export_dir = tmp_path / "exports"
    exported = _payload(
        ops.import_or_export(
            "export",
            str(export_dir),
            object_type="dataset",
            object_ids=["run_001", "run_002"],
        )
    )
    assert exported["count"] == 2
    assert sorted(exported["object_ids"]) == sorted(
        [first_dataset["id"], second_dataset["id"]]
    )
    assert len(exported["paths"]) == 2
    assert all(Path(path).is_file() for path in exported["paths"])

    deleted = _payload(ops.delete("ensemble", "campaign"))
    assert deleted["id"] == ensemble["id"]

    remaining_ensembles = _payload(ops.find("ensemble"))
    assert remaining_ensembles["count"] == 0
    assert remaining_ensembles["returned"] == 0

    remaining_datasets = _payload(ops.find("dataset"))
    assert remaining_datasets["count"] == 2
    assert remaining_datasets["returned"] == 2


def test_kosh_mcp_find_count_mode_uses_fast_path_and_blocks_unsupported_counts(
    tmp_path,
):
    ops = _ops(tmp_path)
    _payload(
        ops.create_dataset_or_ensemble(
            "dataset",
            name="run_001",
            metadata={"status": "new"},
        )
    )
    _payload(
        ops.create_dataset_or_ensemble(
            "dataset",
            name="run_002",
            metadata={"status": "ready"},
        )
    )

    broad_count = _payload(ops.find("dataset", mode="count"))
    assert broad_count["mode"] == "count"
    assert broad_count["execution_path"] == "sql_fast_path"
    assert broad_count["count"] == 2
    assert broad_count["count_method"] == "sql_record_type"

    filtered_count = _payload(
        ops.find(
            "dataset",
            data_criteria={"status": "new"},
            mode="count",
        )
    )
    assert filtered_count["mode"] == "count"
    assert filtered_count["execution_path"] == "sql_fast_path"
    assert filtered_count["count"] == 1
    assert filtered_count["count_method"] == "sql_data_criteria"

    _payload(
        ops.associate_or_dissociate_file(
            "dataset",
            "run_001",
            "a.dat",
            mime_types="text/plain",
            absolute_path=False,
        )
    )

    blocked_count = _payload(
        ops.find(
            "source",
            file_uri="a.dat",
            mode="count",
        )
    )
    assert blocked_count["status"] == "requires_expensive_scan"
    assert blocked_count["mode"] == "count"
    assert blocked_count["execution_path"] == "blocked_expensive_scan"

    allowed_count = _payload(
        ops.find(
            "source",
            file_uri="a.dat",
            mode="count",
            allow_expensive=True,
        )
    )
    assert allowed_count["count"] == 1
    assert allowed_count["execution_path"] == "kosh_find"
    assert allowed_count["count_method"] == "kosh_scan"


def test_kosh_mcp_find_count_mode_supports_file_uri_and_ensemble_data_filters(tmp_path):
    ops = _ops(tmp_path)
    _payload(
        ops.create_dataset_or_ensemble(
            "dataset",
            name="run_001",
            metadata={"shot": 12, "status": "new"},
        )
    )
    _payload(
        ops.create_dataset_or_ensemble(
            "dataset",
            name="run_002",
            metadata={"shot": 30, "status": "ready"},
        )
    )
    _payload(
        ops.create_dataset_or_ensemble(
            "ensemble",
            name="campaign",
            metadata={"stage": "screening"},
        )
    )

    _payload(ops.update("dataset", "run_001", add_to_ensemble="campaign"))
    _payload(ops.update("dataset", "run_002", add_to_ensemble="campaign"))
    _payload(
        ops.associate_or_dissociate_file(
            "dataset",
            "run_001",
            "run_001.dat",
            mime_types="text/plain",
            absolute_path=False,
        )
    )
    _payload(
        ops.associate_or_dissociate_file(
            "ensemble",
            "campaign",
            "campaign.txt",
            mime_types="text/plain",
            absolute_path=False,
        )
    )

    dataset_file_count = _payload(ops.find("dataset", file_uri="run_001.dat", mode="count"))
    assert dataset_file_count["count"] == 1
    assert dataset_file_count["execution_path"] == "sql_fast_path"
    assert dataset_file_count["count_method"] == "sql_file_uri"

    ensemble_file_count = _payload(ops.find("ensemble", file_uri="campaign.txt", mode="count"))
    assert ensemble_file_count["count"] == 1
    assert ensemble_file_count["execution_path"] == "sql_fast_path"
    assert ensemble_file_count["count_method"] == "sql_file_uri"

    ensemble_filtered_count = _payload(
        ops.find(
            "dataset",
            ensemble_id="campaign",
            data_criteria={"shot": {"min": 10, "max": 20}},
            mode="count",
        )
    )
    assert ensemble_filtered_count["count"] == 1
    assert ensemble_filtered_count["execution_path"] == "sql_fast_path"
    assert ensemble_filtered_count["count_method"] == "sql_ensemble_membership_data_criteria"


def test_kosh_mcp_operations_cover_import_and_high_value_failures(tmp_path):
    source_ops = _ops(tmp_path, "source.sqlite")
    source_dataset = _payload(
        source_ops.create_dataset_or_ensemble(
            "dataset",
            name="run_001",
            metadata={"shot": 12},
        )
    )
    export_path = tmp_path / "run_001.json"
    _payload(
        source_ops.import_or_export(
            "export",
            str(export_path),
            object_type="dataset",
            object_ids=source_dataset["id"],
        )
    )

    target_ops = _ops(tmp_path, "target.sqlite")
    imported = _payload(
        target_ops.import_or_export(
            "import",
            str(export_path),
            match_attributes=["name"],
            merge_handler="preserve",
        )
    )
    assert imported["count"] == 1
    assert imported["objects"][0]["name"] == "run_001"
    assert "associated_file_count" not in imported["objects"][0]
    assert "associated_files" not in imported["objects"][0]

    assert "Unknown object_type" in _failure(target_ops.find("not-a-real-type"))
    assert "received 1 MIME type(s) for 2 file URI(s)" in _failure(
        target_ops.associate_or_dissociate_file(
            "dataset",
            "run_001",
            ["a.dat", "b.dat"],
            mime_types=["text/plain"],
        )
    )
    assert "Source identity fields cannot be updated" in _failure(
        target_ops.update("source", "missing-source", attributes={"uri": "new.dat"})
    )
    assert "x_values and y_values must contain the same number of values" in _failure(
        target_ops.add_curve("run_001", "bad_curve", [1, 2], [1])
    )


def test_kosh_mcp_operations_cover_additional_query_load_dataframe_and_export_branches(
    tmp_path,
):
    ops = _ops(tmp_path)

    dataset = _payload(
        ops.create_dataset_or_ensemble(
            "dataset",
            name="run_001",
            metadata={"shot": 12, "status": "new"},
        )
    )
    other_dataset = _payload(
        ops.create_dataset_or_ensemble(
            "dataset",
            name="run_002",
            metadata={"shot": 13, "status": "ready"},
        )
    )
    ensemble = _payload(
        ops.create_dataset_or_ensemble(
            "ensemble",
            name="campaign",
            metadata={"stage": "screening"},
        )
    )

    json_path = tmp_path / "values.json"
    json_path.write_text(
        json.dumps({"A": "a", "B": "b", "C": "c"}),
        encoding="utf-8",
    )
    associated = _payload(
        ops.associate_or_dissociate_file(
            "dataset",
            "run_001",
            str(json_path),
            mime_types="json",
            absolute_path=False,
        )
    )
    dataset_source_id = associated["source_ids"][0]

    ensemble_path = tmp_path / "ensemble.txt"
    ensemble_path.write_text("ensemble source", encoding="utf-8")
    ensemble_associated = _payload(
        ops.associate_or_dissociate_file(
            "ensemble",
            "campaign",
            str(ensemble_path),
            mime_types="text/plain",
            absolute_path=False,
        )
    )
    ensemble_source_id = ensemble_associated["source_ids"][0]

    found_datasets = _payload(ops.find("dataset", ids=["run_001", other_dataset["id"]]))
    assert found_datasets["mode"] == "objects"
    assert found_datasets["returned"] == 2
    assert sorted(result["id"] for result in found_datasets["results"]) == sorted(
        [dataset["id"], other_dataset["id"]]
    )
    assert all("ensembles" not in result for result in found_datasets["results"])

    found_source = _payload(ops.find("source", ids=dataset_source_id))
    assert found_source["returned"] == 1
    assert found_source["results"][0]["id"] == dataset_source_id

    ensemble_sources = _payload(
        ops.find(
            "source",
            owner_type="ensemble",
            owner_id="campaign",
            file_uri=str(ensemble_path),
        )
    )
    assert ensemble_sources["returned"] == 1
    assert ensemble_sources["results"][0]["id"] == ensemble_source_id
    assert ensemble_sources["results"][0]["uri"] == str(ensemble_path)

    _payload(ops.update("dataset", "run_001", add_to_ensemble="campaign"))
    removed = _payload(
        ops.update(
            "dataset",
            "run_001",
            remove_from_ensemble="campaign",
        )
    )
    assert removed["operations"] == [f"removed from ensemble '{ensemble['id']}'"]

    remaining_members = _payload(ops.find("dataset", ensemble_id="campaign"))
    assert remaining_members["count"] == 0
    assert remaining_members["returned"] == 0

    features = _payload(
        ops.load(
            "run_001",
            action="get",
            list_features=True,
            use_cache=False,
        )
    )
    assert features["action"] == "list_features"
    assert features["dataset_id"] == dataset["id"]
    assert {"A", "B", "C", "content"}.issubset(set(features["features"]))

    loaded = _payload(ops.load("run_001", feature_name=["B", "A"]))
    assert loaded["feature_name"] == ["B", "A"]
    assert loaded["data"] == ["b", "a"]

    grouped = _payload(
        ops.load(
            "run_001",
            feature_name=["B", "A"],
            output_format="dict",
            group=True,
        )
    )
    assert grouped["group"] is True
    assert grouped["data"] == {"B": "b", "A": "a"}

    preview = _payload(
        ops.dataframe(
            "preview",
            "dataset",
            object_id="run_001",
            data_columns=["shot", "status"],
        )
    )
    assert preview["scope"] == "dataset"
    assert preview["object_id"] == dataset["id"]
    assert preview["dataframe"]["kind"] == "dataframe"
    assert {"shot", "status"}.issubset(set(preview["columns"]))

    saved = _payload(
        ops.dataframe(
            "save",
            "dataset",
            object_id="run_001",
            output_path=str(tmp_path / "run_001.csv"),
            output_format="csv",
            data_columns=["shot", "status"],
        )
    )
    assert saved["output_format"] == "csv"
    assert Path(saved["path"]).is_file()

    exported = _payload(
        ops.import_or_export(
            "export",
            str(tmp_path / "campaign_export.json"),
            object_type="ensemble",
            object_ids="campaign",
        )
    )
    assert exported["object_ids"] == [ensemble["id"]]
    assert Path(exported["paths"][0]).is_file()


def test_kosh_mcp_operations_cover_multi_input_transform_and_operator_paths(
    tmp_path,
    monkeypatch,
):
    _register_test_operator(monkeypatch)
    ops = _ops(tmp_path)

    first_dataset = _payload(
        ops.create_dataset_or_ensemble("dataset", name="run_001")
    )
    second_dataset = _payload(
        ops.create_dataset_or_ensemble("dataset", name="run_002")
    )

    _payload(
        ops.add_curve(
            "run_001",
            "signal",
            x_values=[0.0, 1.0, 2.0],
            y_values=[10.0, 11.0, 12.0],
            x_label="time",
            y_label="temperature",
        )
    )
    _payload(
        ops.add_curve(
            "run_002",
            "signal",
            x_values=[0.0, 1.0, 2.0],
            y_values=[1.0, 2.0, 3.0],
            x_label="time",
            y_label="temperature",
        )
    )

    inputs = [
        {"dataset_id": "run_001", "feature_name": "signal/temperature"},
        {"dataset_id": "run_002", "feature_name": "signal/temperature"},
    ]
    transformed = _payload(ops.transform(inputs=inputs, output_format="numpy"))
    assert [entry["dataset_id"] for entry in transformed["inputs"]] == [
        first_dataset["id"],
        second_dataset["id"],
    ]
    assert len(transformed["data"]) == 2
    assert transformed["data"][0]["data"] == [10.0, 11.0, 12.0]
    assert transformed["data"][1]["data"] == [1.0, 2.0, 3.0]

    operated = _payload(
        ops.transform(
            inputs=inputs,
            operator="Add",
            output_format="numpy",
        )
    )
    assert operated["operator"] == {"name": "Add", "options": {}}
    assert operated["data"]["data"] == [11.0, 13.0, 15.0]
