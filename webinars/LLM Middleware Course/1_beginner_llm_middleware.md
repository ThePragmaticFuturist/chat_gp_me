# Building a Node.js Middleware Server for LLM Integration

## Course Overview
This course will teach you how to create a Node.js server that acts as middleware between a Large Language Model (LLM) server and a client application. You'll learn about handling file uploads, routing requests, managing sessions, and implementing error handling.

## Prerequisites
- Basic JavaScript knowledge
- Understanding of what an API is
- Familiarity with using a terminal/command line
- A text editor (like Visual Studio Code, Sublime Text, or similar)

## Module 1: Setting Up Your Development Environment

### Step 1: Installing Node.js
1. Remove any existing Node.js installation:
```bash
sudo apt-get remove --purge nodejs nodejs-doc libnode-dev
sudo apt-get autoremove
```

2. Clean up and update package manager:
```bash
sudo apt-get clean
sudo rm -rf /var/lib/apt/lists/*
sudo apt-get update
```

3. Install Node.js 22.x:
```bash
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt-get install -y nodejs
```

4. Set up global npm directory:
```bash
mkdir ~/.npm-global
npm config set prefix '~/.npm-global'
echo 'export PATH=~/.npm-global/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

5. Verify installation:
```bash
node --version
npm --version
```

### Step 2: Project Setup
1. Create project directory:
```bash
mkdir node_projects
cd node_projects
mkdir server
cd server
```

2. Initialize Node.js project:
```bash
npm init -y
```

3. Install required dependencies:
```bash
npm install express axios cors multer form-data
```

## Module 2: Understanding the Core Components

### Express.js Basics
Express.js is a web framework for Node.js that simplifies creating web servers. Key concepts:
- Routes: Define endpoints for your API
- Middleware: Functions that process requests
- Request handling: Managing incoming data
- Response formatting: Structuring outgoing data

### Important Dependencies
1. axios: Makes HTTP requests to the LLM server
   - Used for forwarding requests
   - Handles responses and errors
   
2. cors: Manages Cross-Origin Resource Sharing
   - Allows requests from different domains
   - Controls allowed methods and headers

3. multer: Handles file uploads
   - Manages file storage
   - Validates file types and sizes

4. form-data: Creates multipart form data
   - Used for file uploads
   - Maintains proper formatting

## Module 3: Building the Server

### Step 1: Basic Server Setup
Create `node_server_3000.js` with initial configuration:
```javascript
const express = require('express');
const app = express();
app.use(express.json());

// Basic route
app.get('/', (req, res) => {
  res.send('Server is working!');
});

app.listen(3000, '0.0.0.0', () => {
  console.log('Server is running on port 3000');
});
```

### Step 2: Adding File Upload Functionality
Implement file upload handling with multer:
```javascript
const multer = require('multer');
const storage = multer.diskStorage({
  destination: './uploads',
  filename: (req, file, cb) => {
    cb(null, `${Date.now()}-${file.originalname}`);
  }
});

const upload = multer({ storage: storage });
```

### Step 3: Implementing Error Handling
Add error middleware:
```javascript
app.use((err, req, res, next) => {
  console.error('Server error:', err);
  res.status(500).json({
    error: 'Internal server error',
    message: err.message
  });
});
```

## Module 4: Advanced Features

### Chat Session Management
Learn how to:
- Create chat sessions
- Maintain conversation history
- Handle session timeouts
- Implement cleanup procedures

### File Operations
Understanding:
- Upload validation
- Temporary storage
- Cleanup procedures
- Error handling

### Request Forwarding
Master:
- Proper header handling
- Timeout configuration
- Error recovery
- Response formatting

## Common Issues and Troubleshooting

### Connection Issues
- Verify LLM server is running
- Check network connectivity
- Validate port configurations
- Review CORS settings

### File Upload Problems
- Confirm directory permissions
- Validate file size limits
- Check file type restrictions
- Monitor disk space

### Memory Management
- Watch for memory leaks
- Clean up temporary files
- Monitor resource usage
- Implement proper error handling

## Best Practices and Production Considerations

### Security
1. Input validation
2. Rate limiting
3. Error handling
4. Secure file operations

### Performance
1. Request timeout handling
2. Resource cleanup
3. Logging implementation
4. Error recovery

### Maintenance
1. Code organization
2. Documentation
3. Error logging
4. Monitoring setup

## Next Steps
After completing this course:
1. Implement additional security measures
2. Add monitoring and logging
3. Optimize performance
4. Add user authentication
5. Implement rate limiting
6. Add request validation

## Resources
- Express.js documentation: https://expressjs.com/
- Multer documentation: https://github.com/expressjs/multer
- Node.js best practices: https://nodejs.org/en/docs/guides/
- Error handling guide: https://expressjs.com/en/guide/error-handling.html
