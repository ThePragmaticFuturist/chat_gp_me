// node_server_3000.js

/**
 * This server acts as middleware between an LLM (Large Language Model) server and a client application.
 * It handles file uploads, message routing, and manages communication between the front-end and the AI backend.
 * 
 * Key responsibilities:
 * - File upload handling and management
 * - Request routing to LLM server
 * - Error handling and response formatting
 * - Session management for chat functionality
 * - Cross-Origin Resource Sharing (CORS) configuration
 */

const express = require('express');  // Web framework for Node.js
const axios = require('axios');      // HTTP client for making requests to the LLM server
const cors = require('cors');        // Middleware to enable cross-origin requests
const multer = require('multer');    // Middleware for handling file uploads
const FormData = require('form-data');// Library for creating form data (used in file uploads)
const fs = require('fs');            // Node.js file system module
const path = require('path');        // Utility for working with file paths
const app = express();               // Create Express application instance

/**
 * Configure multer for handling file uploads
 * This sets up where files will be stored and how they'll be named
 */
const storage = multer.diskStorage({
  // Set the destination folder for uploaded files
  destination: (req, file, cb) => {
    // Create uploads directory if it doesn't exist
    if (!fs.existsSync('uploads')) {
      fs.mkdirSync('uploads');
    }
    cb(null, 'uploads/');
  },
  // Generate unique filenames using timestamps
  filename: (req, file, cb) => {
    cb(null, `${Date.now()}-${file.originalname}`);
  },
});

/**
 * Configure upload settings including:
 * - File size limits
 * - Allowed file types
 * - Storage location
 */
const upload = multer({
  storage: storage,
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB size limit
  },
  fileFilter: (req, file, cb) => {
    const allowedTypes = ['application/pdf', 'text/plain'];
    if (allowedTypes.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error('Invalid file type. Only PDF and TXT files are allowed.'));
    }
  },
});

/**
 * Configure CORS settings to allow cross-origin requests
 * In production, origin should be restricted to specific domains
 */
app.use(cors({
  origin: '*',  
  methods: ['GET', 'POST', 'DELETE'],
  allowedHeaders: ['Content-Type']
}));

// Enable JSON parsing for incoming requests
app.use(express.json());

// LLM server address - modify this to match your setup
const localhost = "http://192.168.50.120:5000";

/**
 * Debug route logging - prints all registered routes on startup
 * Useful for debugging and verifying route configuration
 */
app._router.stack.forEach(function(r){
    if (r.route && r.route.path){
        console.log(`Registered route: ${Object.keys(r.route.methods)} ${r.route.path}`);
    }
});

/**
 * Clear database endpoint
 * Allows clearing of the LLM server's database
 */
app.get('/clear_db', async (req, res) => {
  const response = await axios.get(`${localhost}/clear_db`, 
      { 
        headers: { 'Content-Type': 'application/json' },
        timeout: 900000  // 15 minute timeout for long operations
      });

  console.log(response);
  res.json(response.data);
});

/**
 * File upload endpoint
 * Handles file uploads from the client and forwards them to the LLM server
 * Includes error handling and cleanup of temporary files
 */
app.post('/upload', upload.single('file'), async (req, res) => {
  try {
    // Verify file presence
    if (!req.file) {
      console.error('No file in request');
      return res.status(400).json({ error: 'No file uploaded' });
    }

    // Log file details for debugging
    console.log('File received:', {
      filename: req.file.originalname,
      mimetype: req.file.mimetype,
      size: req.file.size
    });

    // Create form data for the LLM server
    const formData = new FormData();
    formData.append('file', fs.createReadStream(req.file.path), {
      filename: req.file.originalname,
      contentType: req.file.mimetype,
    });

    // Forward file to LLM server
    console.log('Forwarding to Python server:', `${localhost}/upload`);
    
    const response = await axios.post(`${localhost}/upload`, formData, {
      headers: {
        ...formData.getHeaders(),
      },
      maxContentLength: Infinity,
      maxBodyLength: Infinity,
      timeout: 900000, // 15 minute timeout
    });

    // Clean up temporary file
    console.log('Upload successful, cleaning up temp file');
    try {
      fs.unlinkSync(req.file.path);
    } catch (cleanupError) {
      console.error('Error cleaning up temp file:', cleanupError);
    }

    res.json(response.data);
  } catch (error) {
    console.error('Upload error:', error);
    res.status(500).json({ 
      error: error.message,
      details: error.response?.data || 'No additional details available'
    });
  }
});


/**
 * Document text retrieval endpoint
 * Fetches specific text chunks from the LLM server based on provided IDs
 * Used for retrieving context during RAG (Retrieval Augmented Generation)
 */
app.post('/get_texts', async (req, res) => {
  try {
    const { ids } = req.body;
    
    console.log('Requesting texts for IDs:', ids);

    // Validate input parameters
    if (!ids || !Array.isArray(ids) || ids.length === 0) {
      console.error('Invalid request body:', req.body);
      return res.status(400).json({ 
        error: 'Invalid or missing document IDs',
        details: 'The request must include an array of document IDs'
      });
    }

    // Forward request to LLM server
    const response = await axios.post(
      `${localhost}/get_texts`,
      { ids },
      { 
        headers: { 'Content-Type': 'application/json' },
        timeout: 900000  // 15 minute timeout
      }
    );
    
    console.log('Received response from Python server:', {
      status: response.status,
      documentCount: response.data.count
    });
    
    res.json(response.data);
  } catch (error) {
    // Detailed error logging and handling
    console.error('Error fetching texts:', {
      message: error.message,
      code: error.code,
      response: error.response?.data,
      status: error.response?.status
    });
    
    // Return appropriate error responses based on error type
    if (error.code === 'ECONNREFUSED') {
      res.status(503).json({ 
        error: 'Service unavailable',
        details: 'Cannot connect to AI service'
      });
    } else if (error.code === 'ETIMEDOUT') {
      res.status(504).json({ 
        error: 'Gateway timeout',
        details: 'AI service took too long to respond'
      });
    } else {
      res.status(error.response?.status || 500).json({ 
        error: 'Failed to fetch document chunks',
        details: error.response?.data?.error || error.message
      });
    }
  }
});

/**
 * Delete specific text chunk endpoint
 * Removes individual chunks of text from the LLM server's database
 * Used for managing and curating the knowledge base
 */
app.delete('/delete_chunk/:id', async (req, res) => {
  try {
    const { id } = req.params;
    
    const response = await axios.delete(
      `${localhost}/delete_chunk/${id}`,
      { timeout: 900000 }
    );
    
    console.log('Delete chunk response:', response.data);
    res.json(response.data);
  } catch (error) {
    console.error('Error deleting chunk:', error);
    res.status(500).json({ 
      error: error.message,
      details: error.response?.data || 'No additional details available'
    });
  }
});

/**
 * Delete entire document endpoint
 * Removes complete documents and their associated chunks from the database
 * Used for managing the document collection
 */
app.delete('/delete_document', express.json(), async (req, res) => {
  try {
    const { ids } = req.body;
    
    // Validate input
    if (!ids || !Array.isArray(ids)) {
      console.log('Invalid request body:', req.body);
      return res.status(400).json({ error: 'Invalid or missing document IDs' });
    }

    console.log('Attempting to delete documents with IDs:', ids);

    // Forward deletion request to LLM server
    const response = await axios.delete(`${localhost}/delete_document`, {
      data: { ids },
      headers: { 'Content-Type': 'application/json' }
    });
    
    console.log('Delete document response:', response.data);
    res.json(response.data);
  } catch (error) {
    // Detailed error logging
    console.error('Delete document error:', {
      message: error.message,
      code: error.code,
      response: error.response?.data,
      status: error.response?.status,
      body: req.body
    });

    res.status(error.response?.status || 500).json({
      error: 'Delete operation failed',
      details: error.response?.data || error.message
    });
  }
});

/**
 * Question answering endpoint
 * Handles direct questions to the LLM, optionally using RAG
 * Main endpoint for getting AI responses
 */
app.get('/ask', async (req, res) => {
  try {
    const { text, use_rag } = req.query;
    console.log('Making request to:', `${localhost}/generate`);
    
    // Forward question to LLM server
    const response = await axios.post(
      `${localhost}/generate`,
      { text, use_rag: Boolean(use_rag) },
      { 
        headers: { 'Content-Type': 'application/json' },
        timeout: 900000
      }
    );
    
    console.log('Received response:', response.data);
    res.json(response.data);
  } catch (error) {
    // Detailed error logging
    console.error('Detailed error:', {
      message: error.message,
      code: error.code,
      response: error.response?.data,
      status: error.response?.status
    });
    
    // Handle specific error cases
    if (error.code === 'ECONNREFUSED') {
      res.status(503).json({ error: 'Service unavailable - Cannot connect to AI service' });
    } else if (error.code === 'ETIMEDOUT') {
      res.status(504).json({ error: 'Gateway timeout - AI service took too long to respond' });
    } else {
      res.status(500).json({ 
        error: error.message,
        details: error.response?.data || 'No additional details available'
      });
    }
  }
});

/**
 * Chat Session Management Endpoints
 * These endpoints handle the creation, management, and cleanup of chat sessions
 */

/**
 * Start new chat session
 * Creates a new conversation session with initial message
 */
app.post('/chat/start', async (req, res) => {
  try {
    const { sessionId, initialMessage } = req.body;
    
    const response = await axios.post(
      `${localhost}/chat/start`,
      { sessionId, message: initialMessage },
      { 
        headers: { 'Content-Type': 'application/json' },
        timeout: 900000
      }
    );
    
    res.json(response.data);
  } catch (error) {
    console.error('Chat start error:', error);
    res.status(500).json({ 
      error: error.message,
      details: error.response?.data || 'No additional details available'
    });
  }
});

/**
 * Send message in existing chat
 * Handles individual messages within an existing chat session
 */
app.post('/chat/message', async (req, res) => {
  try {
    const { sessionId, message } = req.body;
    
    const response = await axios.post(
      `${localhost}/chat/message`,
      { sessionId, message },
      { 
        headers: { 'Content-Type': 'application/json' },
        timeout: 900000
      }
    );
    
    res.json(response.data);
  } catch (error) {
    console.error('Chat message error:', error);
    res.status(500).json({ 
      error: error.message,
      details: error.response?.data || 'No additional details available'
    });
  }
});

/**
 * Retrieve chat history
 * Gets the complete conversation history for a specific session
 */
app.get('/chat/history/:sessionId', async (req, res) => {
  try {
    const { sessionId } = req.params;
    
    const response = await axios.get(
      `${localhost}/chat/history/${sessionId}`,
      { 
        headers: { 'Content-Type': 'application/json' },
        timeout: 900000
      }
    );
    
    res.json(response.data);
  } catch (error) {
    console.error('Chat history error:', error);
    res.status(500).json({ 
      error: error.message,
      details: error.response?.data || 'No additional details available'
    });
  }
});

/**
 * Delete chat session
 * Removes a chat session and its associated history
 */
app.delete('/chat/session/:sessionId', async (req, res) => {
  try {
    const { sessionId } = req.params;
    
    const response = await axios.delete(
      `${localhost}/chat/session/${sessionId}`,
      { 
        headers: { 'Content-Type': 'application/json' },
        timeout: 900000
      }
    );
    
    res.json(response.data);
  } catch (error) {
    console.error('Chat session deletion error:', error);
    res.status(500).json({ 
      error: error.message,
      details: error.response?.data || 'No additional details available'
    });
  }
});

/**
 * Settings management endpoint
 * Handles LLM configuration settings
 */
app.get('/settings', async (req, res) => {
  try {
    const response = await axios.post(
      `${localhost}/settings`, 
      req.query,
      { 
        headers: { 'Content-Type': 'application/json' },
        timeout: 900000
      }
    );
    
    console.log('Received response:', response.data);
    res.json(response.data);
  } catch (error) {
    // Detailed error logging
    console.error('Detailed error:', {
      message: error.message,
      code: error.code,
      response: error.response?.data,
      status: error.response?.status
    });
    
    // Handle different types of errors
    if (error.code === 'ECONNREFUSED') {
      res.status(503).json({ error: 'Service unavailable - Cannot connect to AI service' });
    } else if (error.code === 'ETIMEDOUT') {
      res.status(504).json({ error: 'Gateway timeout - AI service took too long to respond' });
    } else {
      res.status(500).json({ 
        error: error.message,
        details: error.response?.data || 'No additional details available'
      });
    }
  }
});

/**
 * Available models endpoint
 * Retrieves list of available LLM models from the server
 */
app.get('/models', async (req, res) => {
  try {
    const response = await axios.get(
      `${localhost}/models`,
      { 
        headers: { 'Content-Type': 'application/json' },
        timeout: 900000
      }
    );
    
    console.log('Received response:', response.data);
    res.json(response.data);
  } catch (error) {
    console.error('Detailed error:', {
      message: error.message,
      code: error.code,
      response: error.response?.data,
      status: error.response?.status
    });
    
    if (error.code === 'ECONNREFUSED') {
      res.status(503).json({ error: 'Service unavailable - Cannot connect to AI service' });
    } else if (error.code === 'ETIMEDOUT') {
      res.status(504).json({ error: 'Gateway timeout - AI service took too long to respond' });
    } else {
      res.status(500).json({ 
        error: error.message,
        details: error.response?.data || 'No additional details available'
      });
    }
  }
});

/**
 * Basic health check endpoint
 * Used to verify server is running
 */
app.get('/', (req, res) => {
  res.send('Server is working!');
});

/**
 * Global error handling middleware
 * Catches any unhandled errors in the application
 */
app.use((err, req, res, next) => {
  console.error('Server error:', err);
  res.status(500).json({
    error: 'Internal server error',
    message: err.message
  });
});

/**
 * Start the server
 * Listen on all network interfaces (0.0.0.0) on port 3000
 */
app.listen(3000, '0.0.0.0', () => {
  console.log('Server is running on port 3000');
});
