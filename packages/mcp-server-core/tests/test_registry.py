from __future__ import annotations

import pytest

from mcp_server_core.registry import (
    ToolRegistrationError,
    register_directory,
    register_package,
)
from mcp_server_core.server import build_server
from mcp_server_core.settings import ServerSettings


class FakeServer:
    """Records registrations without needing a real MCPServer."""

    def __init__(self) -> None:
        self.registered: list[str] = []

    def tool(self, *args, **kwargs):  # noqa: ANN002, ANN003
        def decorator(fn):  # noqa: ANN001
            self.registered.append(fn.__name__)
            return fn

        return decorator


def write(directory, name: str, body: str):
    path = directory / name
    path.write_text(body)
    return path


TOOL_MODULE = """
def register(server):
    @server.tool()
    def alpha() -> str:
        '''Alpha tool.'''
        return "a"
"""


class TestRegisterDirectory:
    def test_loads_tool_module(self, tmp_path):
        write(tmp_path, "mytools.py", TOOL_MODULE)
        server = FakeServer()
        assert register_directory(server, tmp_path) == ["mytools"]
        assert server.registered == ["alpha"]

    def test_missing_directory_is_not_an_error(self, tmp_path):
        """A project may legitimately have no tools and rely on retrieval."""
        assert register_directory(FakeServer(), tmp_path / "absent") == []

    def test_skips_underscore_prefixed_modules(self, tmp_path):
        write(tmp_path, "_helpers.py", TOOL_MODULE)
        assert register_directory(FakeServer(), tmp_path) == []

    def test_module_without_register_is_skipped(self, tmp_path):
        write(tmp_path, "notatool.py", "VALUE = 1\n")
        assert register_directory(FakeServer(), tmp_path) == []

    def test_import_error_is_fatal(self, tmp_path):
        """A project deployed without the tools it declares is a silently
        broken assistant -- worse than a failed boot."""
        write(tmp_path, "broken.py", "import a_module_that_does_not_exist\n")
        with pytest.raises(ToolRegistrationError, match="broken"):
            register_directory(FakeServer(), tmp_path)

    def test_register_raising_is_fatal(self, tmp_path):
        write(tmp_path, "bad.py", "def register(server):\n    raise ValueError('nope')\n")
        with pytest.raises(ToolRegistrationError, match="register"):
            register_directory(FakeServer(), tmp_path)

    def test_non_callable_register_is_fatal(self, tmp_path):
        write(tmp_path, "weird.py", "register = 42\n")
        with pytest.raises(ToolRegistrationError, match="not callable"):
            register_directory(FakeServer(), tmp_path)

    def test_same_filename_in_two_projects_does_not_collide(self, tmp_path):
        """Both projects have tools.py. Without per-directory module naming the
        second import silently reuses the first's cached module."""
        a = tmp_path / "alpha" / "tools"
        b = tmp_path / "beta" / "tools"
        a.mkdir(parents=True)
        b.mkdir(parents=True)
        write(a, "tools.py", TOOL_MODULE)
        write(
            b,
            "tools.py",
            "def register(server):\n"
            "    @server.tool()\n"
            "    def beta_only() -> str:\n"
            "        '''Beta tool.'''\n"
            "        return 'b'\n",
        )
        server = FakeServer()
        register_directory(server, a)
        register_directory(server, b)
        assert server.registered == ["alpha", "beta_only"]


class TestRegisterPackage:
    def test_loads_builtin_tools(self):
        server = FakeServer()
        loaded = register_package(server, "mcp_server_core.tools")
        assert "mcp_server_core.tools.diagnostics" in loaded
        assert {"ping", "server_info", "fetch_url"} <= set(server.registered)

    def test_unknown_package_is_fatal(self):
        with pytest.raises(ToolRegistrationError, match="cannot import"):
            register_package(FakeServer(), "no_such_package_xyz")


class TestBuildServer:
    def test_builds_with_builtin_tools(self):
        server = build_server(ServerSettings(project=""))
        assert server is not None

    def test_registers_project_tools(self, tmp_path):
        tools = tmp_path / "demo" / "tools"
        tools.mkdir(parents=True)
        write(tools, "extra.py", TOOL_MODULE)
        server = build_server(ServerSettings(project="demo", projects_dir=tmp_path))
        assert server is not None

    def test_missing_project_tools_is_not_fatal(self, tmp_path):
        """Warn, don't fail: retrieval-only projects are legitimate."""
        assert build_server(
            ServerSettings(project="ghost", projects_dir=tmp_path)
        ) is not None
