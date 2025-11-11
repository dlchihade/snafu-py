#!/bin/bash

# Docker MCP Client Setup Script for Cursor
# This script sets up the Docker MCP client connection to Cursor

set -e

echo "🚀 Setting up Docker MCP Client for Cursor..."
echo "=============================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop first."
    exit 1
fi

echo "✅ Docker is running"

# Create .cursor directory if it doesn't exist
mkdir -p .cursor

# Connect Docker MCP client to Cursor
echo "🔗 Connecting Docker MCP client to Cursor..."
if docker mcp client connect cursor; then
    echo "✅ Docker MCP client connected to Cursor successfully"
else
    echo "❌ Failed to connect Docker MCP client to Cursor"
    exit 1
fi

# Install MCP filesystem server globally (optional)
echo "📦 Installing MCP filesystem server (optional)..."
if command -v npm > /dev/null 2>&1; then
    npm install -g @modelcontextprotocol/server-filesystem
    echo "✅ MCP filesystem server installed"
else
    echo "⚠️  npm not found. MCP filesystem server not installed."
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Restart Cursor to activate the MCP connection"
echo "2. The Docker MCP client is now connected and ready to use"
echo "3. You can use Docker commands through Cursor's AI assistant"
echo ""
echo "Configuration files created:"
echo "- .cursor/mcp-config.json (Cursor MCP configuration)"
echo "- mcp-config.json (General MCP configuration)"
echo "- docker-compose.mcp.yml (Docker Compose setup)"
echo ""
echo "To start MCP services with Docker Compose:"
echo "  docker-compose -f docker-compose.mcp.yml up -d"
echo ""
echo "To stop MCP services:"
echo "  docker-compose -f docker-compose.mcp.yml down"
