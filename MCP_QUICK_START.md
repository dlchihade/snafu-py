# MCP Servers Quick Start Guide

## 🚀 Overview

Your MCP (Model Context Protocol) servers are now set up and ready to run! You have three MCP servers configured:

1. **Filesystem Server** - Access and manage local files
2. **Outlook Email Server** - Search, read, and analyze Outlook emails with response assistance
3. **Docker Server** (optional) - Manage Docker containers

## 📧 Outlook Server Setup (IMPORTANT)

Before using the Outlook server, you MUST configure Azure AD credentials:

### Step 1: Register Azure AD Application

1. Go to [Azure Portal](https://portal.azure.com/#blade/Microsoft_AAD_RegisteredApps/ApplicationsListBlade)
2. Click "New registration"
3. Name: "MCP Outlook Server" (or your choice)
4. Select "Accounts in this organizational directory only"
5. Click "Register"

### Step 2: Configure Permissions

1. In your app registration, go to "API permissions"
2. Click "Add a permission" > "Microsoft Graph" > "Application permissions"
3. Add these permissions:
   - `Mail.Read` - Read mail in all mailboxes
   - `Mail.ReadWrite` - Read and write mail in all mailboxes  
   - `User.Read.All` - Read all users' profiles
4. **Click "Grant admin consent"** (requires admin privileges)

### Step 3: Create Client Secret

1. Go to "Certificates & secrets"
2. Click "New client secret"
3. Add description, select expiry
4. **Copy the secret value immediately** (won't be shown again!)

### Step 4: Configure Environment

Edit the file `/Users/diettachihade/snafu-py/mcp-outlook-server/.env`:

```bash
AZURE_TENANT_ID=your-tenant-id-here
AZURE_CLIENT_ID=your-app-client-id-here  
AZURE_CLIENT_SECRET=your-secret-here
OUTLOOK_USER_ID=your-email@company.com
```

Find these values in Azure Portal:
- **Tenant ID**: Azure Active Directory > Overview
- **Client ID**: Your app registration > Overview > Application (client) ID
- **Client Secret**: The value you copied in Step 3
- **Outlook User ID**: The email address you want to access

## 🏃 Running the Servers

### Option 1: Easy Script (Recommended)

```bash
cd /Users/diettachihade/snafu-py
./run-mcp-servers.sh
```

Choose option 1 for Docker or option 2 for local Node.js.

### Option 2: Docker Compose

```bash
cd /Users/diettachihade/snafu-py
docker-compose -f docker-compose.mcp.yml up -d
```

To view logs:
```bash
docker-compose -f docker-compose.mcp.yml logs -f
```

To stop:
```bash
docker-compose -f docker-compose.mcp.yml down
```

### Option 3: Run Locally

```bash
# Terminal 1: Filesystem Server
npx -y @modelcontextprotocol/server-filesystem /Users/diettachihade/snafu-py

# Terminal 2: Outlook Server
cd /Users/diettachihade/snafu-py/mcp-outlook-server
npm start
```

## 🔧 Available Outlook Server Tools

Once running, you can use these tools through your MCP client:

### 1. **search_emails**
Search emails by keywords, sender, subject:
```json
{
  "query": "from:boss@company.com subject:urgent",
  "maxResults": 25
}
```

### 2. **read_email**
Read full email content:
```json
{
  "messageId": "AAMkAGI2..."
}
```

### 3. **analyze_email**
Get sentiment analysis and response suggestions:
```json
{
  "messageId": "AAMkAGI2...",
  "includeConversation": true
}
```

### 4. **get_recent_emails**
Get recent emails from inbox:
```json
{
  "folder": "inbox",
  "count": 10
}
```

### 5. **get_conversation**
Get entire email thread:
```json
{
  "conversationId": "AAQkAGI2...",
  "maxMessages": 10
}
```

### 6. **mark_as_read**
Mark emails as read/unread:
```json
{
  "messageId": "AAMkAGI2...",
  "isRead": true
}
```

## 📝 Email Analysis Features

The Outlook server provides intelligent email analysis:

- **Sentiment Detection**: Identifies positive, negative, or neutral tone
- **Action Items**: Extracts tasks and requests from emails
- **Key Topics**: Identifies main discussion points
- **Response Suggestions**: Provides context-appropriate reply templates
- **Conversation Threading**: Analyzes entire email threads for context

## 🔍 Example Workflows

### Deep Email Search
"Search for all emails from John about the project proposal from last week"

### Email Response Assistance
"Analyze this email and suggest professional responses considering the conversation history"

### Inbox Management
"Show me high-priority unread emails from today and mark them as read after review"

## 🛠 Troubleshooting

### Azure AD Issues
- Verify tenant ID and client ID are correct
- Ensure admin consent is granted
- Check client secret hasn't expired

### Connection Issues
- Ensure Docker is running
- Check firewall allows HTTPS to graph.microsoft.com
- Verify environment variables are set correctly

### Server Not Starting
- Check logs: `docker-compose -f docker-compose.mcp.yml logs mcp-outlook-server`
- Verify `.env` file exists and is configured
- Ensure Node.js version 18+ is installed (for local running)

## 🔒 Security Notes

- **Never commit `.env` files to Git**
- Store Azure credentials securely
- Rotate client secrets regularly
- Use least-privilege permissions
- Consider Azure Key Vault for production

## 📚 Additional Resources

- [MCP Documentation](https://modelcontextprotocol.org)
- [Microsoft Graph API Docs](https://docs.microsoft.com/en-us/graph/)
- [Azure AD App Registration Guide](https://docs.microsoft.com/en-us/azure/active-directory/develop/)

## Need Help?

Check the detailed README in:
- `/Users/diettachihade/snafu-py/mcp-outlook-server/README.md`

Or view server logs:
```bash
docker-compose -f docker-compose.mcp.yml logs -f
```
