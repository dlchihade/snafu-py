#!/bin/bash

# MCP Servers Runner Script
# This script sets up and runs all MCP servers

set -e  # Exit on any error

echo "🚀 Starting MCP Servers Setup and Execution"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}docker-compose is not available. Please install it first.${NC}"
    exit 1
fi

# Function to check if .env file exists for Outlook server
check_outlook_env() {
    if [ ! -f "mcp-outlook-server/.env" ]; then
        echo -e "${YELLOW}⚠️  Outlook server .env file not found${NC}"
        echo "Creating .env file from template..."
        
        if [ -f "mcp-outlook-server/env.example" ]; then
            cp mcp-outlook-server/env.example mcp-outlook-server/.env
            echo -e "${YELLOW}Please edit mcp-outlook-server/.env with your Azure AD credentials${NC}"
            echo "Refer to mcp-outlook-server/README.md for setup instructions"
            echo ""
            read -p "Press Enter once you've configured the .env file, or Ctrl+C to exit..."
        else
            echo -e "${RED}env.example file not found in mcp-outlook-server${NC}"
            exit 1
        fi
    fi
}

# Function to run with docker-compose
run_with_docker() {
    echo -e "${GREEN}Running MCP servers with Docker...${NC}"
    
    # Check for Outlook env configuration
    check_outlook_env
    
    # Source the Outlook .env file for docker-compose
    if [ -f "mcp-outlook-server/.env" ]; then
        export $(cat mcp-outlook-server/.env | grep -v '^#' | xargs)
    fi
    
    # Use docker-compose or docker compose based on availability
    if command -v docker-compose &> /dev/null; then
        docker-compose -f docker-compose.mcp.yml up -d
    else
        docker compose -f docker-compose.mcp.yml up -d
    fi
    
    echo -e "${GREEN}✅ MCP servers are starting...${NC}"
    echo ""
    echo "Services running:"
    echo "  - MCP Docker Server: http://localhost:3000"
    echo "  - MCP Filesystem Server: http://localhost:3001"
    echo "  - MCP Outlook Server: http://localhost:3002"
    echo ""
    echo "To view logs:"
    echo "  docker-compose -f docker-compose.mcp.yml logs -f"
    echo ""
    echo "To stop all servers:"
    echo "  docker-compose -f docker-compose.mcp.yml down"
}

# Function to run locally (without Docker)
run_locally() {
    echo -e "${GREEN}Running MCP servers locally...${NC}"
    
    # Check Node.js installation
    if ! command -v node &> /dev/null; then
        echo -e "${RED}Node.js is not installed. Please install Node.js first.${NC}"
        exit 1
    fi
    
    # Check npm installation
    if ! command -v npm &> /dev/null; then
        echo -e "${RED}npm is not installed. Please install npm first.${NC}"
        exit 1
    fi
    
    echo "Installing dependencies for Outlook server..."
    cd mcp-outlook-server
    
    # Check for .env file
    if [ ! -f ".env" ]; then
        if [ -f "env.example" ]; then
            cp env.example .env
            echo -e "${YELLOW}Please edit mcp-outlook-server/.env with your Azure AD credentials${NC}"
            echo "Refer to README.md for setup instructions"
            read -p "Press Enter once you've configured the .env file, or Ctrl+C to exit..."
        fi
    fi
    
    npm install
    npm run build
    
    cd ..
    
    echo -e "${GREEN}Starting MCP servers...${NC}"
    
    # Start filesystem server in background
    echo "Starting Filesystem Server..."
    npx -y @modelcontextprotocol/server-filesystem /Users/diettachihade/snafu-py &
    FILESYSTEM_PID=$!
    
    # Start Outlook server in background
    echo "Starting Outlook Server..."
    cd mcp-outlook-server
    npm start &
    OUTLOOK_PID=$!
    cd ..
    
    echo -e "${GREEN}✅ MCP servers are running locally${NC}"
    echo ""
    echo "Server PIDs:"
    echo "  - Filesystem Server: $FILESYSTEM_PID"
    echo "  - Outlook Server: $OUTLOOK_PID"
    echo ""
    echo "To stop servers, use: kill $FILESYSTEM_PID $OUTLOOK_PID"
    
    # Wait for interrupt
    echo "Press Ctrl+C to stop all servers..."
    wait
}

# Main menu
echo "How would you like to run the MCP servers?"
echo "1) Docker (Recommended)"
echo "2) Local Node.js"
echo "3) Exit"
echo ""
read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        run_with_docker
        ;;
    2)
        run_locally
        ;;
    3)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice. Please run the script again.${NC}"
        exit 1
        ;;
esac
