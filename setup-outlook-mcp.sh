#!/bin/bash

# Setup script for MCP Outlook Server
# This helps configure Azure AD credentials

echo "================================================"
echo "   MCP Outlook Server Configuration Setup      "
echo "================================================"
echo ""

ENV_FILE="/Users/diettachihade/snafu-py/mcp-outlook-server/.env"

# Check if .env exists
if [ ! -f "$ENV_FILE" ]; then
    echo "Creating .env file from template..."
    cp /Users/diettachihade/snafu-py/mcp-outlook-server/env.example "$ENV_FILE"
fi

echo "Let's configure your Azure AD credentials for Outlook access."
echo ""
echo "You'll need these from Azure Portal:"
echo "  1. Azure Tenant ID"
echo "  2. Azure Client ID (Application ID)"
echo "  3. Client Secret"
echo "  4. Your Outlook email address"
echo ""
echo "If you haven't set up Azure AD yet, follow the guide at:"
echo "https://portal.azure.com/#blade/Microsoft_AAD_RegisteredApps/ApplicationsListBlade"
echo ""

read -p "Do you have your Azure credentials ready? (y/n): " ready

if [ "$ready" != "y" ] && [ "$ready" != "Y" ]; then
    echo ""
    echo "Please get your Azure credentials first, then run this script again."
    echo "See the guide in: /Users/diettachihade/snafu-py/MCP_QUICK_START.md"
    exit 0
fi

echo ""
read -p "Enter your Azure Tenant ID: " tenant_id
read -p "Enter your Azure Client ID: " client_id
read -s -p "Enter your Client Secret (hidden): " client_secret
echo ""
read -p "Enter your Outlook email address: " email

# Update the .env file
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed -i '' "s/AZURE_TENANT_ID=.*/AZURE_TENANT_ID=$tenant_id/" "$ENV_FILE"
    sed -i '' "s/AZURE_CLIENT_ID=.*/AZURE_CLIENT_ID=$client_id/" "$ENV_FILE"
    sed -i '' "s/AZURE_CLIENT_SECRET=.*/AZURE_CLIENT_SECRET=$client_secret/" "$ENV_FILE"
    sed -i '' "s/OUTLOOK_USER_ID=.*/OUTLOOK_USER_ID=$email/" "$ENV_FILE"
else
    # Linux
    sed -i "s/AZURE_TENANT_ID=.*/AZURE_TENANT_ID=$tenant_id/" "$ENV_FILE"
    sed -i "s/AZURE_CLIENT_ID=.*/AZURE_CLIENT_ID=$client_id/" "$ENV_FILE"
    sed -i "s/AZURE_CLIENT_SECRET=.*/AZURE_CLIENT_SECRET=$client_secret/" "$ENV_FILE"
    sed -i "s/OUTLOOK_USER_ID=.*/OUTLOOK_USER_ID=$email/" "$ENV_FILE"
fi

echo ""
echo "✅ Configuration saved!"
echo ""
echo "Now let's test the setup..."
echo ""

# Test by building
cd /Users/diettachihade/snafu-py/mcp-outlook-server
npm install > /dev/null 2>&1
npm run build

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo ""
    echo "Your MCP Outlook Server is ready to run!"
    echo ""
    echo "To start all MCP servers, run:"
    echo "  cd /Users/diettachihade/snafu-py"
    echo "  ./run-mcp-servers.sh"
    echo ""
    echo "Choose option 2 (Local Node.js) for best results."
else
    echo "❌ Build failed. Please check your Node.js installation."
fi
