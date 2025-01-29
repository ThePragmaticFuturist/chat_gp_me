# Document Processing and RAG: Understanding Modern AI Document Interaction

## Introduction

Imagine you're at a library with a very knowledgeable assistant. You hand the assistant some documents, and they can not only read and understand these documents but also answer questions about them intelligently. This is exactly what we're building with our Document Processing system using RAG (Retrieval-Augmented Generation). 

## What is RAG?

Before we dive into the code, let's understand what RAG is and why it's important. RAG stands for Retrieval-Augmented Generation. Think of it as giving your AI assistant a personalized knowledge base. When you ask a question:

1. The system first searches through your uploaded documents for relevant information (Retrieval)
2. It then combines this specific information with its general knowledge to generate an answer (Augmented Generation)

This approach makes responses more accurate and relevant to your specific documents.

## Building Our Document Processing System

### Part 1: Setting Up the File Upload System

First, let's create the interface for uploading documents. We'll use Expo's DocumentPicker to handle file selection.

```typescript
import * as DocumentPicker from 'expo-document-picker';
import axios from 'axios';

interface UploadedFile {
  id: string;           // Unique identifier for the file
  name: string;         // Original filename
  timestamp: string;    // Upload time
  documentIds: string[]; // IDs of chunks in the database
}

function DocumentProcessor() {
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [uploading, setUploading] = useState(false);

  const handleUpload = async () => {
    try {
      setUploading(true);
      
      // Open document picker - notice we limit to PDF and text files
      const result = await DocumentPicker.getDocumentAsync({
        type: ['application/pdf', 'text/plain'],
        copyToCacheDirectory: true,
      });

      // User cancelled selection
      if (result.canceled) {
        return;
      }

      const file = result.assets[0];
      
      // Create FormData for upload
      let formData = new FormData();
      
      // Handle different platforms (web vs native)
      if (file.uri.startsWith('data:')) {
        // Handle web platform base64 data
        const blob = await fetch(file.uri).then(r => r.blob());
        formData.append('file', blob, file.name);
      } else {
        // Handle native platform file
        formData.append('file', {
          uri: Platform.OS === 'web' ? file.uri : file.uri.replace('file://', ''),
          type: file.mimeType || 'application/octet-stream',
          name: file.name,
        } as any);
      }

      // Send file to server
      const uploadResponse = await axios.post(`${localhost}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      // Create new file record
      const newFile: UploadedFile = {
        id: Date.now().toString(),
        name: file.name,
        timestamp: new Date().toLocaleString(),
        documentIds: uploadResponse.data.document_ids,
      };

      // Update state and storage
      const updatedFiles = [...uploadedFiles, newFile];
      setUploadedFiles(updatedFiles);
      await AsyncStorage.setItem('uploadedFiles', JSON.stringify(updatedFiles));

    } catch (error) {
      Alert.alert('Error', 'Failed to upload file');
    } finally {
      setUploading(false);
    }
  };
}
```

### Part 2: Managing Document Chunks

When we upload a document, our server breaks it into smaller, manageable chunks. Let's create an interface to view and manage these chunks.

```typescript
interface DocumentChunk {
  id: string;         // Unique chunk identifier
  content: string;    // Chunk content
  metadata: any;      // Additional information about the chunk
}

function ChunkManager() {
  const [documentChunks, setDocumentChunks] = useState<DocumentChunk[]>([]);
  const [loadingChunks, setLoadingChunks] = useState(false);

  // Function to load chunks for a specific document
  const loadDocumentChunks = async (file: UploadedFile) => {
    setLoadingChunks(true);
    try {
      const response = await axios.post(`${localhost}/get_texts`, {
        ids: file.documentIds
      });

      if (response.data.documents) {
        setDocumentChunks(response.data.documents);
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to fetch document chunks');
    } finally {
      setLoadingChunks(false);
    }
  };

  // Function to delete a specific chunk
  const handleDeleteChunk = async (chunkId: string) => {
    try {
      await axios.delete(`${localhost}/delete_chunk/${chunkId}`);
      
      // Update UI after deletion
      setDocumentChunks(prevChunks => 
        prevChunks.filter(chunk => chunk.id !== chunkId)
      );
      
    } catch (error) {
      Alert.alert('Error', 'Failed to delete chunk');
    }
  };
}
```

### Part 3: Implementing RAG Queries

Now let's implement the ability to ask questions about our documents:

```typescript
function DocumentQuerying() {
  const [text, setText] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    if (!text.trim()) {
      Alert.alert('Error', 'Please enter a question');
      return;
    }

    setLoading(true);
    try {
      // Note the use_rag flag to enable RAG processing
      const result = await axios.get(`${localhost}/ask`, {
        params: {
          text: text.trim(),
          use_rag: true,
        }
      });
      
      setResponse(result.data.response);
      
      // Save to history with RAG flag
      await saveToHistory(text, result.data.response, modelName, true);
      
    } catch (error) {
      Alert.alert('Error', 'Failed to get response');
    } finally {
      setLoading(false);
    }
  };
}
```

### Part 4: Building the User Interface

Let's create a responsive interface that shows both the document list and query interface:

```typescript
return (
  <SafeAreaView style={styles.container}>
    <KeyboardAvoidingView
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      style={styles.keyboardAvoid}
    >
      <ScrollView
        style={styles.content}
        contentContainerStyle={styles.contentContainer}
      >
        {/* Upload Section */}
        <View style={styles.uploadSection}>
          <View style={styles.buttonRow}>
            <TouchableOpacity
              style={styles.uploadButton}
              onPress={handleUpload}
              disabled={uploading}
            >
              {uploading ? (
                <ActivityIndicator color="#fff" />
              ) : (
                <>
                  <FontAwesome name="upload" size={20} color="#fff" />
                  <Text style={styles.uploadButtonText}>
                    Upload Document
                  </Text>
                </>
              )}
            </TouchableOpacity>
          </View>

          {/* Document List */}
          {uploadedFiles.length > 0 && (
            <View style={styles.filesContainer}>
              <Text style={styles.filesTitle}>Uploaded Documents:</Text>
              {uploadedFiles.map((file) => (
                <TouchableOpacity
                  key={file.id}
                  style={styles.fileItem}
                  onPress={() => handleDocumentPress(file)}
                >
                  <FontAwesome name="file-text-o" size={16} color="#666" />
                  <Text style={styles.fileName}>{file.name}</Text>
                  <Text style={styles.fileTimestamp}>
                    {file.timestamp}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          )}
        </View>

        {/* Query Section */}
        <View style={styles.chatSection}>
          <TextInput
            style={styles.input}
            value={text}
            onChangeText={setText}
            placeholder="Ask about your documents..."
            multiline
          />

          <TouchableOpacity
            style={styles.button}
            onPress={handleSubmit}
            disabled={loading}
          >
            {loading ? (
              <ActivityIndicator color="#fff" />
            ) : (
              <Text style={styles.buttonText}>Ask</Text>
            )}
          </TouchableOpacity>

          {response && (
            <View style={styles.responseContainer}>
              <Text style={styles.responseTitle}>Answer:</Text>
              <ScrollView style={styles.responseScroll}>
                <Text style={styles.responseText}>{response}</Text>
              </ScrollView>
            </View>
          )}
        </View>
      </ScrollView>
    </KeyboardAvoidingView>
  </SafeAreaView>
);
```

## Understanding the Flow

Let's walk through how everything works together:

1. **Document Upload**
   - User selects a document
   - Document is sent to server
   - Server processes document into chunks
   - Chunks are stored in vector database
   - Document metadata is saved locally

2. **Document Management**
   - Users can view uploaded documents
   - Each document can be expanded to see chunks
   - Individual chunks can be deleted
   - Entire documents can be removed

3. **RAG Query Process**
   - User asks a question
   - Question is sent to server with RAG flag
   - Server:
     - Searches for relevant chunks
     - Combines chunks with question
     - Generates response using LLM
   - Response is displayed to user

## Practice Exercises

To better understand the system, try implementing these features:

1. Add support for more document types (e.g., DOCX, XLSX)
2. Implement a preview feature for uploaded documents
3. Add the ability to edit chunk content
4. Create a visualization of chunk relationships
5. Add a feature to combine multiple chunks

## Common Challenges and Solutions

1. **Large File Handling**
   - Implement file size checks
   - Add progress indicators for uploads
   - Consider chunk upload for very large files

2. **Performance Issues**
   - Implement pagination for document lists
   - Use lazy loading for chunk content
   - Cache frequently accessed documents

3. **Error Handling**
   - Implement retry logic for failed uploads
   - Add meaningful error messages
   - Handle network connectivity issues

## Advanced Topics

Once you're comfortable with the basics, consider exploring:

1. **Vector Database Optimization**
   - Understanding embedding algorithms
   - Tuning chunk sizes
   - Optimizing search parameters

2. **Enhanced RAG Techniques**
   - Implementing hybrid search
   - Adding metadata filtering
   - Using different chunking strategies

3. **User Experience Improvements**
   - Adding drag-and-drop upload
   - Implementing document categories
   - Adding search within documents

## Conclusion

Document processing with RAG is a powerful way to give AI systems access to specific knowledge while maintaining context. The key points to remember are:

1. Proper file handling is crucial for reliability
2. Chunk management affects response quality
3. User interface should be intuitive and responsive
4. Error handling must be comprehensive
5. Performance optimization is important for larger documents

Keep experimenting and refining your implementation to create a more powerful and user-friendly system!
