"""MADA extension manifest for Kosh-provided MCP servers."""

from kosh.current_version import current_version


def get_extension_manifest():
    """Return the MADA extension manifest for Kosh.

    :returns: The Kosh extension manifest with its MCP server registration.
    :raises RuntimeError: If the optional ``kosh[mcp]`` dependencies are not
        installed.
    """
    try:
        from mada_tools.extensions import ExtensionManifest, MCPServerRegistration
    except ImportError as e:
        raise RuntimeError(
            "Kosh MADA extension discovery requires the optional 'mcp' extra. "
            "Install it with: pip install 'kosh[mcp]'"
        ) from e

    return ExtensionManifest(
        display_name="Kosh",
        version=current_version,
        provider_package="kosh",
        mcp_servers=(
            MCPServerRegistration(
                name="kosh",
                module_path="kosh.mcp_server.server",
                package="kosh",
                description="MCP server for Kosh data management.",
            ),
        ),
    )
