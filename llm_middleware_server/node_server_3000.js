// node_server_3000.js

const express = require('express');
const axios = require('axios');
const cors = require('cors');
const multer = require('multer');
const FormData = require('form-data');
const fs = require('fs');
const path = require('path');
const app = express();

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    if (!fs.existsSync('uploads')) {
      fs.mkdirSync('uploads');
    }
    cb(null, 'uploads/');
  },
  filename: (req, file, cb) => {
    cb(null, `${Date.now()}-${file.originalname}`);
  },
});

const upload = multer({
  storage: storage,
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB limit
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

app.use(cors({
  origin: '*',  // Should be more restrictive in production
  methods: ['GET', 'POST', 'DELETE'],
  allowedHeaders: ['Content-Type']
}));

app.use(express.json());

const localhost = "http://192.168.50.120:5000";

app._router.stack.forEach(function(r){
    if (r.route && r.route.path){
        console.log(`Registered route: ${Object.keys(r.route.methods)} ${r.route.path}`);
    }
});

app.get('/clear_db', async (req, res) => {
  const response = await axios.get(`${localhost}/clear_db`, 
      { 
        headers: { 'Content-Type': 'application/json' },
        timeout: 900000
      });

  console.log(response);

  res.json(response.data);
});

// File upload endpoint
app.post('/upload', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      console.error('No file in request');
      return res.status(400).json({ error: 'No file uploaded' });
    }

    console.log('File received:', {
      filename: req.file.originalname,
      mimetype: req.file.mimetype,
      size: req.file.size
    });

    const formData = new FormData();
    formData.append('file', fs.createReadStream(req.file.path), {
      filename: req.file.originalname,
      contentType: req.file.mimetype,
    });

    console.log('Forwarding to Python server:', `${localhost}/upload`);
    
    const response = await axios.post(`${localhost}/upload`, formData, {
      headers: {
        ...formData.getHeaders(),
      },
      maxContentLength: Infinity,
      maxBodyLength: Infinity,
      timeout: 900000, // 30 second timeout
    });

    console.log('Upload successful, cleaning up temp file');
    
    // Clean up uploaded file
    try {
      fs.unlinkSync(req.file.path);
    } catch (cleanupError) {
      console.error('Error cleaning up temp file:', cleanupError);
      // Continue even if cleanup fails
    }

    res.json(response.data);
  } catch (error) {
    console.error('Upload error:', error);
    res.status(500).json({ 
      error: error.message,
      details: error.response?.data || 'No additional details available'
    });
  } finally {
    if (req.file?.path) {
      try {
        fs.unlinkSync(req.file.path);
      } catch (cleanupError) {
        console.error('Cleanup error:', cleanupError);
      }
    }
  }
});

// Modified ask endpoint to support RAG
// Update the get_texts endpoint
app.post('/get_texts', async (req, res) => {
  try {
    const { ids } = req.body;
    
    console.log('Requesting texts for IDs:', ids);

    if (!ids || !Array.isArray(ids) || ids.length === 0) {
      console.error('Invalid request body:', req.body);
      return res.status(400).json({ 
        error: 'Invalid or missing document IDs',
        details: 'The request must include an array of document IDs'
      });
    }

    const response = await axios.post(
      `${localhost}/get_texts`,
      { ids },
      { 
        headers: { 'Content-Type': 'application/json' },
        timeout: 900000
      }
    );
    
    console.log('Received response from Python server:', {
      status: response.status,
      documentCount: response.data.count
    });
    
    res.json(response.data);
  } catch (error) {
    console.error('Error fetching texts:', {
      message: error.message,
      code: error.code,
      response: error.response?.data,
      status: error.response?.status
    });
    
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

// Add endpoints for deleting chunks and documents
app.delete('/delete_chunk/:id', async (req, res) => {
  try {
    const { id } = req.params;
    
    const response = await axios.delete(
      `${localhost}/delete_chunk/${id}`,
      { 
        timeout: 900000
      }
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

app.delete('/delete_document', express.json(), async (req, res) => {

  //console.log("req", JSON.stringify(req, null, 2));
  //console.log("res", JSON.stringify(res, null, 2));

  try {
    const { ids } = req.body;
    
    if (!ids || !Array.isArray(ids)) {
      console.log('Invalid request body:', req.body);
      return res.status(400).json({ error: 'Invalid or missing document IDs' });
    }

    console.log('Attempting to delete documents with IDs:', ids);

    const response = await axios.delete(`${localhost}/delete_document`, {
      data: { ids },
      headers: { 'Content-Type': 'application/json' }
    });
    
    console.log('Delete document response:', response.data);
    res.json(response.data);
  } catch (error) {
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

// Modified ask endpoint to support RAG
app.get('/ask', async (req, res) => {
  try {
    const { text, use_rag } = req.query;
    //console.log('Received query:', { text, Boolean(use_rag) });
    console.log('Making request to:', `${localhost}/generate`);
    
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

// Add these new endpoints to node_server_3000.js

// New chat session endpoints
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

// Settings endpoint
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

// Models endpoint
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

app.get('/', (req, res) => {
  res.send('Server is working!');
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('Server error:', err);
  res.status(500).json({
    error: 'Internal server error',
    message: err.message
  });
});

app.listen(3000, '0.0.0.0', () => {
  console.log('Server is running on port 3000');
});