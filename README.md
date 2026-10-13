![Kosh Logo](share/icons/png/Kosh_Logo_Blue.png)
# Overview

Kosh allows codes to store, query, share data via an easy-to-use Python API. Kosh lies on top of Sina and as a result can use any database backend supported by Sina.

In addition Kosh aims to make data access and sharing as simple as possible.

Via "loaders", "transformers" and "operators" Kosh can access and process data in a consistent fashion, decoupled from data format and location. 

Kosh is a Hindi word that means *treasury*, which is derived from *Kosha*, a Sanskrit word that means container in either a direct or metaphorical sense. A fairly good translation would be *repository*.

# Getting Started


To get the latest public version:

```bash
pip install kosh
```

To get the latest stable, from a cloned repo simply run:

```bash
pip install .
```

Alternatively  add the path to this repo to your `PYTHONPATH` environment variable, or in your code with:

```python
import sys
sys.path.append(path_to_kosh_repo)
```

For more details look into the [installation doc](INSTALL.md)

If you want to use Kosh through Codex, see [the Codex skill install guide](docs/codex_skill.md).

If you want to use Kosh from a MADA agent or another MCP-compatible client,
see the [Kosh MCP server guide](docs/source/users/mcp_server.md).

# First steps

See [this file](docs/source/users/index.md)

# Getting Involved
Kosh is user-oriented, and users' questions, comments, and contributions help guide its evolution. We welcome involvement and feedbacks.

# Contact Info
You can reach our team at kosh-support@llnl.gov.
Kosh main developer can be reached at: doutriaux1@llnl.gov

# Contributing
Contributions should be submitted as a pull request pointing to the develop branch, and must pass Kosh's CI process; to run the same checks locally follow the instructions in the tests directory.

Contributions must be made under the same license as Kosh (see the bottom of this file).

# Release and License
Kosh is distributed under the terms of the MIT license; new contributions must be made under this license.

SPDX-License-Identifier: MIT

``LLNL-CODE-814755``
