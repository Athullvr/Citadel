"""
citadel-predict-mcp: Model Context Protocol (MCP) server for Citadel Predict.
"""

from .server import create_server, main, server

__version__ = "0.1.0"
__all__ = ["server", "create_server", "main"]
