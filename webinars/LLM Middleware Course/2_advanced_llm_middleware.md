# Advanced Node.js Middleware Server Concepts

## Lesson 1: Understanding Asynchronous Operations and Promises

### Introduction to Asynchronous Programming
Imagine you're at a restaurant. When you order food, the waiter doesn't stand at your table waiting for the kitchen to cook your meal. Instead, they take your order, submit it to the kitchen, and continue serving other tables. When your food is ready, they come back to deliver it. This is exactly how asynchronous programming works in Node.js!

### Why Asynchronous Operations Matter
In our middleware server, we're constantly dealing with operations that take time:
- Uploading files to the LLM server
- Waiting for AI model responses
- Reading from and writing to files
- Making database queries

If we handled these operations synchronously (one after another), our server would become very slow and unresponsive.

### Understanding Promises
Let's break down Promises with a practical example from our server:

```javascript
// Example of a Promise-based operation
app.post('/upload', upload.single('file'), async (req, res) => {
  try {
    // This is like placing an order at a restaurant
    const response = await axios.post(`${localhost}/upload`, formData, {
      headers: { ...formData.getHeaders() },
      timeout: 900000
    });
    
    // This code only runs after we get the response
    console.log('Upload successful!');
    res.json(response.data);
  } catch (error) {
    // This handles any problems that occurred
    console.error('Upload failed:', error);
    res.status(500).json({ error: error.message });
  }
});
```

Think of a Promise as a contract that says: "I promise to give you a result, but it might take some time. When I'm done, I'll either give you the result (resolve) or tell you what went wrong (reject)."

### The async/await Pattern
Instead of dealing with .then() chains, we use async/await to make our code more readable:

```javascript
// The old way with .then()
axios.post(url, data)
  .then(response => {
    console.log(response);
  })
  .catch(error => {
    console.error(error);
  });

// The modern way with async/await
async function uploadFile() {
  try {
    const response = await axios.post(url, data);
    console.log(response);
  } catch (error) {
    console.error(error);
  }
}
```

### Practice Exercise
Try converting this Promise-based code to use async/await:

```javascript
function getFileContents(filePath) {
  return fs.readFile(filePath)
    .then(data => {
      return data.toString();
    })
    .catch(error => {
      console.error('Failed to read file:', error);
      throw error;
    });
}
```

## Lesson 2: Managing File Uploads and Cleanup

### The File Upload Process
File uploads in our server follow a specific lifecycle:
1. Receive file from client
2. Store temporarily
3. Process and forward to LLM server
4. Clean up temporary files

### Setting Up Multer
Multer is our file handling middleware. Here's how we configure it:

```javascript
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    // Create uploads directory if it doesn't exist
    if (!fs.existsSync('uploads')) {
      fs.mkdirSync('uploads');
    }
    cb(null, 'uploads/');
  },
  filename: (req, file, cb) => {
    // Generate unique filename
    cb(null, `${Date.now()}-${file.originalname}`);
  }
});

const upload = multer({
  storage: storage,
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB limit
  },
  fileFilter: (req, file, cb) => {
    // Validate file types
    const allowedTypes = ['application/pdf', 'text/plain'];
    if (allowedTypes.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error('Invalid file type'));
    }
  }
});
```

### Proper File Cleanup
Always clean up temporary files, even if an error occurs:

```javascript
app.post('/upload', upload.single('file'), async (req, res) => {
  try {
    // Process file...
    
    // Clean up at the end
    if (req.file?.path) {
      fs.unlinkSync(req.file.path);
    }
  } catch (error) {
    // Clean up even if there's an error
    if (req.file?.path) {
      try {
        fs.unlinkSync(req.file.path);
      } catch (cleanupError) {
        console.error('Cleanup error:', cleanupError);
      }
    }
    res.status(500).json({ error: error.message });
  }
});
```

## Lesson 3: Handling Errors Properly

### The Importance of Error Handling
Good error handling:
- Prevents server crashes
- Provides meaningful feedback
- Helps with debugging
- Maintains security

### Types of Errors
1. Operational Errors (expected):
   - File not found
   - Network timeout
   - Invalid input

2. Programming Errors (bugs):
   - Null reference
   - Type errors
   - Syntax errors

### Implementing Error Handling
We use multiple layers of error handling:

```javascript
// Route-level error handling
app.post('/ask', async (req, res) => {
  try {
    // Validate input
    if (!req.query.text) {
      return res.status(400).json({ 
        error: 'Missing required parameter: text' 
      });
    }

    const response = await axios.post(
      `${localhost}/generate`,
      { text: req.query.text },
      { timeout: 900000 }
    );
    
    res.json(response.data);
  } catch (error) {
    // Specific error handling
    if (error.code === 'ECONNREFUSED') {
      res.status(503).json({ 
        error: 'Service unavailable - Cannot connect to AI service' 
      });
    } else if (error.code === 'ETIMEDOUT') {
      res.status(504).json({ 
        error: 'Gateway timeout - AI service took too long to respond' 
      });
    } else {
      // General error handler
      res.status(500).json({ 
        error: error.message,
        details: error.response?.data || 'No additional details available'
      });
    }
  }
});

// Global error handler
app.use((err, req, res, next) => {
  console.error('Server error:', err);
  res.status(500).json({
    error: 'Internal server error',
    message: err.message
  });
});
```

## Lesson 4: Maintaining Proper Session State

### Understanding Session State
Session state helps us:
- Track conversation history
- Maintain context
- Handle user-specific settings
- Manage resources

### Implementing Chat Sessions
Here's how we manage chat sessions:

```javascript
app.post('/chat/start', async (req, res) => {
  try {
    const { sessionId, initialMessage } = req.body;
    
    // Validate session parameters
    if (!sessionId || !initialMessage) {
      return res.status(400).json({
        error: 'Missing required parameters'
      });
    }
    
    // Start new chat session
    const response = await axios.post(
      `${localhost}/chat/start`,
      { sessionId, message: initialMessage },
      { timeout: 900000 }
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

// Managing session history
app.get('/chat/history/:sessionId', async (req, res) => {
  try {
    const { sessionId } = req.params;
    
    const response = await axios.get(
      `${localhost}/chat/history/${sessionId}`,
      { timeout: 900000 }
    );
    
    res.json(response.data);
  } catch (error) {
    console.error('Chat history error:', error);
    res.status(500).json({ 
      error: error.message 
    });
  }
});

// Cleaning up sessions
app.delete('/chat/session/:sessionId', async (req, res) => {
  try {
    const { sessionId } = req.params;
    
    const response = await axios.delete(
      `${localhost}/chat/session/${sessionId}`,
      { timeout: 900000 }
    );
    
    res.json(response.data);
  } catch (error) {
    console.error('Session deletion error:', error);
    res.status(500).json({ 
      error: error.message 
    });
  }
});
```

### Best Practices for Session Management
1. Always validate session IDs
2. Implement session timeouts
3. Clean up expired sessions
4. Handle disconnections gracefully
5. Maintain proper error handling

### Practice Exercise
Try implementing a simple session timeout mechanism:

```javascript
// Session timeout example (pseudocode)
const sessions = new Map();
const SESSION_TIMEOUT = 30 * 60 * 1000; // 30 minutes

function createSession(sessionId) {
  sessions.set(sessionId, {
    created: Date.now(),
    lastAccessed: Date.now(),
    data: {}
  });
}

function checkSession(sessionId) {
  const session = sessions.get(sessionId);
  if (!session) return false;
  
  const now = Date.now();
  if (now - session.lastAccessed > SESSION_TIMEOUT) {
    sessions.delete(sessionId);
    return false;
  }
  
  session.lastAccessed = now;
  return true;
}
```

## Conclusion
Understanding these four key aspects of server development will help you build robust, reliable middleware servers. Remember to:
- Use async/await for cleaner asynchronous code
- Always clean up resources
- Implement comprehensive error handling
- Maintain proper session state

Practice these concepts by building small test servers and gradually adding more complex functionality as you become comfortable with each aspect.
