# Building a Full-Stack LLM Chat Application: A Complete Guide

This guide will walk you through creating a full-stack application that enables users to interact with a Large Language Model (LLM) through a modern web and mobile interface.

## Prerequisites

Before starting, you'll need to install:
1. Node.js (v16 or later) - Download from nodejs.org
2. Python (v3.8 or later) - Download from python.org
3. Xcode (for iOS development) - Download from the Mac App Store
4. Visual Studio Code or another code editor
5. Git for version control

## Part 1: Setting Up the Development Environment

### Installing Node.js and npm
1. Download Node.js from nodejs.org
2. Run the installer
3. Verify installation:
```bash
node --version
npm --version
```

### Installing Expo CLI
```bash
npm install -g expo-cli
```

### Creating the Project Structure
1. Create a new directory for your project
```bash
mkdir llm-chat-app
cd llm-chat-app
```

2. Create subdirectories for each component:
```bash
mkdir client
mkdir server
```

## Part 2: Setting Up the Client (Expo App)

### Creating the Expo Project
```bash
cd client
npx create-expo-app chatgp_me --template
```

### Installing Required Dependencies
```bash
cd chatgp_me
npm install @react-native-async-storage/async-storage @react-navigation/bottom-tabs @react-navigation/native axios expo-blur expo-constants expo-document-picker expo-font expo-haptics expo-linking expo-router expo-splash-screen expo-status-bar expo-symbols expo-system-ui expo-web-browser react-native-modal react-native-gesture-handler react-native-reanimated react-native-safe-area-context react-native-screens react-native-web react-native-webview
```

### Understanding Key Dependencies
- `expo-router`: Handles navigation and routing in the app
- `@react-navigation/bottom-tabs`: Creates the bottom tab navigation
- `axios`: Makes HTTP requests to the server
- `@react-native-async-storage/async-storage`: Stores data locally on the device
- `react-native-modal`: Creates modal dialogs
- `expo-document-picker`: Allows users to select files

## Part 3: Setting Up the Node.js Server

### Creating the Server
```bash
cd ../server
npm init -y
```

### Installing Server Dependencies
```bash
npm install express cors multer axios body-parser
```

### Understanding Server Components
1. `express`: Web framework for Node.js
2. `cors`: Handles Cross-Origin Resource Sharing
3. `multer`: Processes file uploads
4. `body-parser`: Parses incoming request bodies

### Basic Server Setup
Create `server.js`:
```javascript
const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const multer = require('multer');

const app = express();
app.use(cors());
app.use(bodyParser.json());

const port = 3000;
app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});
```

## Part 4: Understanding the App Architecture

### Client-Side Components
1. Tab Navigation System
   - Home tab (`index.tsx`): Simple question-answer interface
   - Chat tab (`chat.tsx`): Continuous chat interface
   - History tab (`two.tsx`): View past conversations
   - Documents tab (`rag.tsx`): Upload and process documents

2. State Management
   - Uses React Context (`AppContext.tsx`) for global state
   - Manages model selection and settings
   - Handles file uploads and chat history

3. UI Components
   - `AlertModal.tsx`: Reusable modal component
   - `ThemedText.tsx` and `ThemedView.tsx`: Theme-aware components
   - `Collapsible.tsx`: Expandable content sections

### Key Features Implementation

#### Chat System
The chat system (`chat.tsx`) implements:
- Real-time messaging with the LLM
- Chat session management
- Message history
- Sidebar navigation on web
- Responsive design for mobile and web

Key functions:
```javascript
// Starts a new chat session
const startNewChat = async () => {
  // Creates unique session ID
  // Initializes chat in backend
  // Updates local state
}

// Sends message to LLM
const sendMessage = async () => {
  // Validates input
  // Sends to server
  // Updates UI with response
  // Saves to history
}
```

#### Document Processing (`rag.tsx`)
Implements:
- File upload system
- Document chunk management
- RAG (Retrieval-Augmented Generation) queries
- Document deletion and management

Key functions:
```javascript
// Handles document upload
const handleUpload = async () => {
  // Picks document using expo-document-picker
  // Creates FormData
  // Sends to server
  // Updates local state
}

// Processes RAG queries
const handleSubmit = async () => {
  // Sends query with RAG flag
  // Processes response
  // Updates UI
}
```

## Part 5: Running the Application

### Starting the Development Environment
1. Start the Node.js server:
```bash
cd server
node server.js
```

2. Start the Expo development server:
```bash
cd client/chatgp_me
expo start
```

### Accessing the Application
- Web: Open browser to http://localhost:19006
- iOS: 
  - Install Expo Go app on your iPhone
  - Scan QR code from terminal
  - Or run in iOS simulator: `expo start --ios`

## Part 6: Common Issues and Solutions

### CORS Issues
If experiencing CORS errors:
1. Verify server CORS configuration
2. Check API endpoint URLs
3. Ensure proper headers in requests

### File Upload Problems
Common issues:
1. File size limits
2. File type restrictions
3. FormData formation

### State Management
Tips for debugging:
1. Use React Developer Tools
2. Check AsyncStorage data
3. Monitor network requests

## Part 7: Best Practices and Tips

### Code Organization
1. Keep components modular
2. Use consistent naming conventions
3. Implement proper error handling
4. Add detailed comments
5. Follow React/React Native best practices

### Performance Optimization
1. Implement proper loading states
2. Use memory efficiently
3. Optimize image and file handling
4. Implement proper cleanup in useEffect

### Security Considerations
1. Validate all inputs
2. Sanitize file uploads
3. Implement proper error handling
4. Use secure communication protocols
5. Handle sensitive data appropriately

## Next Steps and Resources

### Further Learning
1. React Native Documentation
2. Expo Documentation
3. Node.js Best Practices
4. LLM Integration Patterns

### Community Resources
1. Expo Forums
2. React Native Community
3. Stack Overflow
4. GitHub Discussions

This guide provides a foundation for building the LLM chat application. As you progress, you'll discover more advanced features and optimizations you can implement to enhance the user experience and application functionality.
