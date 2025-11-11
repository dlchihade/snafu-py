# MCP Outlook Server

An MCP (Model Context Protocol) server for Microsoft Outlook email management with deep search, content reading, and response assistance capabilities.

## Features

- **Email Search**: Deep search through emails using keywords, sender, subject, or date ranges
- **Email Reading**: Read full email content including attachments information
- **Conversation Threading**: Access entire email conversation threads
- **Email Analysis**: Analyze emails for sentiment, action items, and key topics
- **Response Assistance**: Get intelligent suggestions for email responses
- **Email Management**: Mark emails as read/unread, access recent emails from specific folders

## Setup

### 1. Azure AD App Registration

1. Go to [Azure Portal](https://portal.azure.com/#blade/Microsoft_AAD_RegisteredApps/ApplicationsListBlade)
2. Click "New registration"
3. Name your app (e.g., "MCP Outlook Server")
4. Select "Accounts in this organizational directory only"
5. Click "Register"

### 2. Configure API Permissions

In your app registration:
1. Go to "API permissions"
2. Click "Add a permission" > "Microsoft Graph"
3. Select "Application permissions" (for server-to-server)
4. Add these permissions:
   - `Mail.Read` - Read mail in all mailboxes
   - `Mail.ReadWrite` - Read and write mail in all mailboxes
   - `User.Read.All` - Read all users' profiles
5. Click "Grant admin consent" (requires admin privileges)

### 3. Create Client Secret

1. Go to "Certificates & secrets"
2. Click "New client secret"
3. Add a description and select expiry
4. Copy the secret value immediately (won't be shown again)

### 4. Configure Environment

Copy `env.example` to `.env` and fill in your values:

```bash
cp env.example .env
```

Edit `.env` with your Azure AD credentials:
- `AZURE_TENANT_ID`: Your Azure AD tenant ID
- `AZURE_CLIENT_ID`: Your app's client ID
- `AZURE_CLIENT_SECRET`: The secret you created
- `OUTLOOK_USER_ID`: Email address to access

### 5. Install Dependencies

```bash
npm install
```

### 6. Build and Run

```bash
# Build TypeScript
npm run build

# Run the server
npm start

# Or run in development mode
npm run dev
```

## Docker Setup

The server is included in the `docker-compose.mcp.yml` file. To run with Docker:

```bash
docker-compose -f docker-compose.mcp.yml up mcp-outlook-server
```

## Usage with MCP Client

Add to your MCP client configuration:

```json
{
  "mcpServers": {
    "outlook": {
      "command": "node",
      "args": ["/path/to/mcp-outlook-server/dist/index.js"],
      "env": {
        "AZURE_TENANT_ID": "your-tenant-id",
        "AZURE_CLIENT_ID": "your-client-id",
        "AZURE_CLIENT_SECRET": "your-secret",
        "OUTLOOK_USER_ID": "user@example.com"
      }
    }
  }
}
```

## Available Tools

### search_emails
Search for emails using various criteria:
```json
{
  "query": "from:john@example.com subject:meeting",
  "maxResults": 25
}
```

### read_email
Read the full content of an email:
```json
{
  "messageId": "AAMkAGI2..."
}
```

### get_conversation
Get all messages in a conversation thread:
```json
{
  "conversationId": "AAQkAGI2...",
  "maxMessages": 10
}
```

### analyze_email
Analyze an email and get response suggestions:
```json
{
  "messageId": "AAMkAGI2...",
  "includeConversation": true
}
```

### get_recent_emails
Get recent emails from a folder:
```json
{
  "folder": "inbox",
  "count": 10
}
```

### mark_as_read
Mark an email as read or unread:
```json
{
  "messageId": "AAMkAGI2...",
  "isRead": true
}
```

## Security Notes

- Store credentials securely
- Use environment variables for sensitive data
- Regularly rotate client secrets
- Follow principle of least privilege for permissions
- Consider using Azure Key Vault for production

## Troubleshooting

### Authentication Errors
- Verify your Azure AD credentials are correct
- Ensure admin consent is granted for permissions
- Check tenant ID and client ID match your app registration

### Permission Errors
- Confirm Mail.Read and Mail.ReadWrite permissions are granted
- Verify admin consent has been provided
- Check the user account has a valid mailbox

### Connection Issues
- Ensure you have internet connectivity
- Check firewall settings allow HTTPS traffic to graph.microsoft.com
- Verify proxy settings if behind corporate network
