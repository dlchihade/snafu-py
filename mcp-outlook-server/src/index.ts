import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  Tool,
} from '@modelcontextprotocol/sdk/types.js';
import { Client } from '@microsoft/microsoft-graph-client';
import { TokenCredentialAuthenticationProvider } from '@microsoft/microsoft-graph-client/authProviders/azureTokenCredentials/index.js';
import { ClientSecretCredential } from '@azure/identity';
import * as dotenv from 'dotenv';
import 'isomorphic-fetch';

dotenv.config();

// Email interface
interface EmailMessage {
  id: string;
  subject: string;
  from: string;
  to: string[];
  receivedDateTime: string;
  bodyPreview: string;
  body?: string;
  hasAttachments: boolean;
  importance: string;
  isRead: boolean;
  conversationId?: string;
}

// Microsoft Graph Client setup
class OutlookClient {
  private client: Client;

  constructor() {
    // Using app-only authentication for server scenarios
    const credential = new ClientSecretCredential(
      process.env.AZURE_TENANT_ID!,
      process.env.AZURE_CLIENT_ID!,
      process.env.AZURE_CLIENT_SECRET!
    );

    const authProvider = new TokenCredentialAuthenticationProvider(credential, {
      scopes: ['https://graph.microsoft.com/.default'],
    });

    this.client = Client.initWithMiddleware({
      authProvider: authProvider,
    });
  }

  async searchEmails(query: string, maxResults: number = 25): Promise<EmailMessage[]> {
    try {
      const userId = process.env.OUTLOOK_USER_ID || 'me';
      const response = await this.client
        .api(`/users/${userId}/messages`)
        .search(query)
        .top(maxResults)
        .select('id,subject,from,toRecipients,receivedDateTime,bodyPreview,hasAttachments,importance,isRead,conversationId')
        .orderby('receivedDateTime desc')
        .get();

      return response.value.map((msg: any) => ({
        id: msg.id,
        subject: msg.subject,
        from: msg.from?.emailAddress?.address || 'Unknown',
        to: msg.toRecipients?.map((r: any) => r.emailAddress?.address) || [],
        receivedDateTime: msg.receivedDateTime,
        bodyPreview: msg.bodyPreview,
        hasAttachments: msg.hasAttachments,
        importance: msg.importance,
        isRead: msg.isRead,
        conversationId: msg.conversationId,
      }));
    } catch (error) {
      console.error('Error searching emails:', error);
      throw error;
    }
  }

  async getEmailContent(messageId: string): Promise<EmailMessage> {
    try {
      const userId = process.env.OUTLOOK_USER_ID || 'me';
      const response = await this.client
        .api(`/users/${userId}/messages/${messageId}`)
        .select('id,subject,from,toRecipients,receivedDateTime,body,bodyPreview,hasAttachments,importance,isRead,conversationId')
        .get();

      return {
        id: response.id,
        subject: response.subject,
        from: response.from?.emailAddress?.address || 'Unknown',
        to: response.toRecipients?.map((r: any) => r.emailAddress?.address) || [],
        receivedDateTime: response.receivedDateTime,
        bodyPreview: response.bodyPreview,
        body: response.body?.content,
        hasAttachments: response.hasAttachments,
        importance: response.importance,
        isRead: response.isRead,
        conversationId: response.conversationId,
      };
    } catch (error) {
      console.error('Error getting email content:', error);
      throw error;
    }
  }

  async getConversation(conversationId: string, maxMessages: number = 10): Promise<EmailMessage[]> {
    try {
      const userId = process.env.OUTLOOK_USER_ID || 'me';
      const response = await this.client
        .api(`/users/${userId}/messages`)
        .filter(`conversationId eq '${conversationId}'`)
        .top(maxMessages)
        .select('id,subject,from,toRecipients,receivedDateTime,bodyPreview,hasAttachments,importance,isRead')
        .orderby('receivedDateTime asc')
        .get();

      return response.value.map((msg: any) => ({
        id: msg.id,
        subject: msg.subject,
        from: msg.from?.emailAddress?.address || 'Unknown',
        to: msg.toRecipients?.map((r: any) => r.emailAddress?.address) || [],
        receivedDateTime: msg.receivedDateTime,
        bodyPreview: msg.bodyPreview,
        hasAttachments: msg.hasAttachments,
        importance: msg.importance,
        isRead: msg.isRead,
      }));
    } catch (error) {
      console.error('Error getting conversation:', error);
      throw error;
    }
  }

  async markAsRead(messageId: string, isRead: boolean = true): Promise<void> {
    try {
      const userId = process.env.OUTLOOK_USER_ID || 'me';
      await this.client
        .api(`/users/${userId}/messages/${messageId}`)
        .patch({ isRead });
    } catch (error) {
      console.error('Error marking email as read:', error);
      throw error;
    }
  }

  async getRecentEmails(folder: string = 'inbox', count: number = 10): Promise<EmailMessage[]> {
    try {
      const userId = process.env.OUTLOOK_USER_ID || 'me';
      const folderPath = folder === 'inbox' ? 'inbox' : folder;
      
      const response = await this.client
        .api(`/users/${userId}/mailFolders/${folderPath}/messages`)
        .top(count)
        .select('id,subject,from,toRecipients,receivedDateTime,bodyPreview,hasAttachments,importance,isRead,conversationId')
        .orderby('receivedDateTime desc')
        .get();

      return response.value.map((msg: any) => ({
        id: msg.id,
        subject: msg.subject,
        from: msg.from?.emailAddress?.address || 'Unknown',
        to: msg.toRecipients?.map((r: any) => r.emailAddress?.address) || [],
        receivedDateTime: msg.receivedDateTime,
        bodyPreview: msg.bodyPreview,
        hasAttachments: msg.hasAttachments,
        importance: msg.importance,
        isRead: msg.isRead,
        conversationId: msg.conversationId,
      }));
    } catch (error) {
      console.error('Error getting recent emails:', error);
      throw error;
    }
  }
}

// Response helper for analyzing and suggesting email responses
class ResponseHelper {
  analyzeEmailContext(email: EmailMessage, conversation?: EmailMessage[]): any {
    const analysis: any = {
      sentiment: this.analyzeSentiment(email.body || email.bodyPreview),
      urgency: email.importance === 'high' ? 'High' : 'Normal',
      requiresAction: this.detectActionItems(email.body || email.bodyPreview),
      keyTopics: this.extractKeyTopics(email.body || email.bodyPreview),
      suggestedTone: 'professional',
    };

    if (conversation && conversation.length > 1) {
      analysis.conversationContext = {
        threadLength: conversation.length,
        isOngoing: true,
        previousExchanges: conversation.length - 1,
      };
    }

    return analysis;
  }

  private analyzeSentiment(text: string): string {
    // Simple sentiment analysis (can be enhanced with NLP libraries)
    const positiveWords = ['thank', 'appreciate', 'great', 'excellent', 'good', 'pleased', 'happy'];
    const negativeWords = ['concern', 'issue', 'problem', 'urgent', 'disappointed', 'unhappy', 'complaint'];
    
    const lower = text.toLowerCase();
    const positiveCount = positiveWords.filter(word => lower.includes(word)).length;
    const negativeCount = negativeWords.filter(word => lower.includes(word)).length;
    
    if (positiveCount > negativeCount) return 'positive';
    if (negativeCount > positiveCount) return 'negative';
    return 'neutral';
  }

  private detectActionItems(text: string): string[] {
    const actionPhrases = [
      /please\s+(\w+\s+){1,5}/gi,
      /could\s+you\s+(\w+\s+){1,5}/gi,
      /would\s+you\s+(\w+\s+){1,5}/gi,
      /need\s+to\s+(\w+\s+){1,5}/gi,
      /must\s+(\w+\s+){1,5}/gi,
      /should\s+(\w+\s+){1,5}/gi,
    ];

    const actions: string[] = [];
    actionPhrases.forEach(pattern => {
      const matches = text.match(pattern);
      if (matches) {
        actions.push(...matches.map(m => m.trim()));
      }
    });

    return [...new Set(actions)].slice(0, 5); // Return unique top 5 action items
  }

  private extractKeyTopics(text: string): string[] {
    // Simple keyword extraction (can be enhanced with NLP)
    const stopWords = new Set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were']);
    const words = text.toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(word => word.length > 3 && !stopWords.has(word));

    const wordFreq = new Map<string, number>();
    words.forEach(word => {
      wordFreq.set(word, (wordFreq.get(word) || 0) + 1);
    });

    return Array.from(wordFreq.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([word]) => word);
  }

  generateResponseSuggestions(analysis: any): string[] {
    const suggestions = [];

    if (analysis.sentiment === 'negative') {
      suggestions.push(
        "Thank you for bringing this to my attention. I understand your concern and will look into it immediately.",
        "I apologize for any inconvenience this may have caused. Let me investigate and get back to you with a solution."
      );
    } else if (analysis.sentiment === 'positive') {
      suggestions.push(
        "Thank you for your message. I'm glad to hear things are going well.",
        "I appreciate your positive feedback. It's great to see the progress we're making."
      );
    }

    if (analysis.requiresAction.length > 0) {
      suggestions.push(
        "I'll take care of the items you've mentioned and provide an update by [date].",
        "I've noted the action items and will begin working on them right away."
      );
    }

    if (suggestions.length === 0) {
      suggestions.push(
        "Thank you for your email. I've reviewed the information and will respond accordingly.",
        "I appreciate you reaching out. Let me review this and get back to you shortly."
      );
    }

    return suggestions;
  }
}

// MCP Server setup
class OutlookMCPServer {
  private server: Server;
  private outlookClient: OutlookClient;
  private responseHelper: ResponseHelper;

  constructor() {
    this.server = new Server(
      {
        name: 'outlook-mcp-server',
        version: '1.0.0',
      },
      {
        capabilities: {
          tools: {},
        },
      }
    );

    this.outlookClient = new OutlookClient();
    this.responseHelper = new ResponseHelper();
    this.setupHandlers();
  }

  private setupHandlers() {
    this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: this.getTools(),
    }));

    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;

      switch (name) {
        case 'search_emails':
          return await this.handleSearchEmails(args);
        case 'read_email':
          return await this.handleReadEmail(args);
        case 'get_conversation':
          return await this.handleGetConversation(args);
        case 'analyze_email':
          return await this.handleAnalyzeEmail(args);
        case 'get_recent_emails':
          return await this.handleGetRecentEmails(args);
        case 'mark_as_read':
          return await this.handleMarkAsRead(args);
        default:
          throw new Error(`Unknown tool: ${name}`);
      }
    });
  }

  private getTools(): Tool[] {
    return [
      {
        name: 'search_emails',
        description: 'Search for emails using keywords, sender, subject, or date range',
        inputSchema: {
          type: 'object',
          properties: {
            query: { type: 'string', description: 'Search query (keywords, sender email, subject, etc.)' },
            maxResults: { type: 'number', description: 'Maximum number of results to return (default: 25)' },
          },
          required: ['query'],
        },
      },
      {
        name: 'read_email',
        description: 'Read the full content of a specific email by ID',
        inputSchema: {
          type: 'object',
          properties: {
            messageId: { type: 'string', description: 'The ID of the email message to read' },
          },
          required: ['messageId'],
        },
      },
      {
        name: 'get_conversation',
        description: 'Get all messages in an email conversation thread',
        inputSchema: {
          type: 'object',
          properties: {
            conversationId: { type: 'string', description: 'The conversation ID' },
            maxMessages: { type: 'number', description: 'Maximum number of messages to retrieve (default: 10)' },
          },
          required: ['conversationId'],
        },
      },
      {
        name: 'analyze_email',
        description: 'Analyze an email for sentiment, action items, and get response suggestions',
        inputSchema: {
          type: 'object',
          properties: {
            messageId: { type: 'string', description: 'The ID of the email to analyze' },
            includeConversation: { type: 'boolean', description: 'Include conversation context in analysis' },
          },
          required: ['messageId'],
        },
      },
      {
        name: 'get_recent_emails',
        description: 'Get recent emails from a specific folder',
        inputSchema: {
          type: 'object',
          properties: {
            folder: { type: 'string', description: 'Folder name (default: inbox)' },
            count: { type: 'number', description: 'Number of emails to retrieve (default: 10)' },
          },
        },
      },
      {
        name: 'mark_as_read',
        description: 'Mark an email as read or unread',
        inputSchema: {
          type: 'object',
          properties: {
            messageId: { type: 'string', description: 'The ID of the email message' },
            isRead: { type: 'boolean', description: 'Mark as read (true) or unread (false)' },
          },
          required: ['messageId'],
        },
      },
    ];
  }

  private async handleSearchEmails(args: any) {
    const { query, maxResults = 25 } = args;
    const emails = await this.outlookClient.searchEmails(query, maxResults);
    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify(emails, null, 2),
        },
      ],
    };
  }

  private async handleReadEmail(args: any) {
    const { messageId } = args;
    const email = await this.outlookClient.getEmailContent(messageId);
    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify(email, null, 2),
        },
      ],
    };
  }

  private async handleGetConversation(args: any) {
    const { conversationId, maxMessages = 10 } = args;
    const conversation = await this.outlookClient.getConversation(conversationId, maxMessages);
    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify(conversation, null, 2),
        },
      ],
    };
  }

  private async handleAnalyzeEmail(args: any) {
    const { messageId, includeConversation = false } = args;
    const email = await this.outlookClient.getEmailContent(messageId);
    
    let conversation: EmailMessage[] | undefined;
    if (includeConversation && email.conversationId) {
      conversation = await this.outlookClient.getConversation(email.conversationId);
    }

    const analysis = this.responseHelper.analyzeEmailContext(email, conversation);
    const suggestions = this.responseHelper.generateResponseSuggestions(analysis);

    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify({
            email: {
              subject: email.subject,
              from: email.from,
              receivedDateTime: email.receivedDateTime,
            },
            analysis,
            responseSuggestions: suggestions,
          }, null, 2),
        },
      ],
    };
  }

  private async handleGetRecentEmails(args: any) {
    const { folder = 'inbox', count = 10 } = args;
    const emails = await this.outlookClient.getRecentEmails(folder, count);
    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify(emails, null, 2),
        },
      ],
    };
  }

  private async handleMarkAsRead(args: any) {
    const { messageId, isRead = true } = args;
    await this.outlookClient.markAsRead(messageId, isRead);
    return {
      content: [
        {
          type: 'text',
          text: `Email ${messageId} marked as ${isRead ? 'read' : 'unread'}`,
        },
      ],
    };
  }

  async start() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error('Outlook MCP Server started');
  }
}

// Start the server
const server = new OutlookMCPServer();
server.start().catch(console.error);
