# Kosh MCP Server

Kosh provides an MCP server so MADA agents and other MCP-compatible clients can query and update a Kosh store through Kosh tools. The simplest workflow is:

1. Install `kosh[mcp]`
2. Define a `kosh` server in `my_servers.json`
3. Start that server with `mada-tools`
4. Point a MADA agent at `http://localhost:<port>/mcp`
5. Run `mada-cli`

## Install

Install Kosh with MCP support:

```bash
pip install "kosh[mcp]"
```

For development from a checkout:

```bash
pip install -e ".[mcp]"
```

After installation, you can confirm that MADA sees the Kosh server extension:

```bash
mada-tools available-servers
```

You should see a server named `kosh`.

## Quick start with MADA

### 1. Create a server configuration

Create a `my_servers.json` file that tells `mada-tools` how to start the Kosh MCP server:

```json
{
  "servers": {
    "kosh": {
      "host": "localhost",
      "port": 8000,
      "transport": "streamable-http",
      "env_vars": {
        "KOSH_DATABASE_URI": "kosh_store.sqlite",
        "KOSH_CONNECTION_TYPE": "write",
        "KOSH_STORE_ALIASES_JSON": "{\"scratch\":\"scratch.sqlite\",\"archive\":\"/data/archive.sqlite\"}",
        "KOSH_LOGLEVEL": "WARNING"
      }
    }
  }
}
```

The values shown above are the defaults. You can omit `host`, `port`,
`transport`, `KOSH_CONNECTION_TYPE`, and `KOSH_LOGLEVEL` when you want default
behavior.

`KOSH_DATABASE_URI` can be a SQLite file path or any database URI supported by
Kosh and Sina. This remains the startup default store when the user does not
switch stores during chat.

If `KOSH_DATABASE_URI` is not set, the server uses `kosh_store.sqlite` in the current working directory.

`KOSH_EXECUTION_OPTIONS` is also supported and may be passed as JSON in `env_vars`:

```bash
export KOSH_EXECUTION_OPTIONS='{"isolation_level": "READ COMMITTED"}'
```

`KOSH_STORE_ALIASES_JSON` is optional and accepts a JSON object mapping
friendly alias names to database URIs:

```bash
export KOSH_STORE_ALIASES_JSON='{"scratch":"scratch.sqlite","archive":"/data/archive.sqlite"}'
```

These aliases stay on the server side so the agent can switch stores with an
alias instead of repeating a full URI.

### 2. Start the server

Start the Kosh MCP server through `mada-tools`:

```bash
mada-tools start-servers my_servers.json
```

When you are finished, stop managed servers with:

```bash
mada-tools stop-servers
```

### 3. Create an agent configuration

Create a `my_agents.json` file for the MADA agent that should use the Kosh tools:

```json
{
  "model": {
    "provider": "livai",
    "model": "gpt-5.4",
    "api_key": "${API_KEY}",
    "base_url": "${API_BASE_URL:-https://livai-api.llnl.gov/v1}"
  },
  "agents": [
    {
      "agent_name": "KoshAgent",
      "description": "Stores, queries, and retrieves data from a Kosh store",
      "domain": "kosh",
      "mcp_servers": ["kosh"],
      "instructions": [
        "You are the Kosh agent. Use the Kosh MCP tools to inspect and update the active store. If the user asks which store is active, call store_connection(action=\"current\"). If the user asks to switch stores, call store_connection(action=\"connect\", ...) first."
      ]
    }
  ],
  "mcp_servers": {
    "kosh": {
      "transport": "streamable-http",
      "url": "http://localhost:8000/mcp",
      "description": "Kosh MCP server"
    }
  },
  "interface": {
    "title": "Kosh Agent",
    "description": "Chat with a MADA agent connected to a Kosh store",
    "chat_placeholder": "Ask about datasets, ensembles, and files...",
    "port": 7860,
    "share": false
  }
}
```

The `kosh` name must match between `my_servers.json` and `my_agents.json`.

### 4. Run the agent

For a command line chat interface:

```bash
mada-cli my_agents.json
```

If you prefer a web interface, use:

```bash
mada-gradio my_agents.json
```

The `interface` block in the example above is only needed for `mada-gradio`.

## Switching stores during chat

The Kosh MCP server keeps the existing startup defaults unless the user asks to
switch stores. Use `store_connection(action="current")` to inspect the current
active store and return its absolute local `full_path` when applicable.

To change the active store for later tool calls, use `store_connection` with
exactly one of:

- `alias`, using an alias from `KOSH_STORE_ALIASES_JSON`
- `database_uri`, using a raw URI or local file path supplied at runtime

Examples:

```text
store_connection(action="current")
store_connection(action="connect", alias="archive")
store_connection(action="connect", database_uri="/tmp/experiment.sqlite")
```

The current `KOSH_CONNECTION_TYPE` and `KOSH_EXECUTION_OPTIONS` stay in effect
after a switch.

If the user switches to a local SQLite path that does not exist yet and the
server is in write mode, Kosh will create that store and `store_connection` will
return a warning describing that creation.

## Advanced: start the server directly

If you want to run the Kosh MCP server without `mada-tools`, set the Kosh store URI and start it directly:

```bash
export KOSH_DATABASE_URI=kosh_store.sqlite
export KOSH_CONNECTION_TYPE=write
export KOSH_STORE_ALIASES_JSON='{"scratch":"scratch.sqlite","archive":"/data/archive.sqlite"}'
export KOSH_LOGLEVEL=WARNING
export KOSH_EXECUTION_OPTIONS='{"isolation_level": "READ COMMITTED"}'
mada-mcp-kosh --transport streamable-http --host localhost --port 8000
```

For transport options, managed server configuration, and server lifecycle commands, see:

- [MADA Tools documentation](https://software.llnl.gov/mada-tools/develop/)
- [MADA Tools GitHub repository](https://github.com/llnl/mada-tools)
