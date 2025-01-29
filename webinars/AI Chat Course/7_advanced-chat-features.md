# Advanced Chat Features: Enhancing User Experience and Functionality

## Introduction

Think about your favorite messaging app. What makes it special? Is it the ability to express yourself with rich text formatting? The ease of sharing files and images? The satisfaction of reacting to messages? Today, we'll learn how to implement these engaging features in our chat application, making it more powerful and user-friendly.

## Part 1: Message Formatting and Markdown Support

Let's begin by implementing rich text formatting. We'll use a markdown parser to transform text markers into styled content.

```typescript
import MarkdownIt from 'markdown-it';
import { StyleSheet } from 'react-native';

class MessageFormatter {
    private md: MarkdownIt;
    
    constructor() {
        // Initialize markdown parser with specific features enabled
        this.md = new MarkdownIt({
            html: false,        // Disable HTML for security
            breaks: true,       // Convert \n to <br>
            linkify: true,      // Detect URLs and convert to links
            typographer: true   // Enable smart quotes and other features
        });
        
        // Add custom plugins for additional formatting
        this.addCustomFormatting();
    }
    
    private addCustomFormatting() {
        // Add support for code highlighting
        this.md.use(require('markdown-it-prism'));
        
        // Add support for task lists
        this.md.use(require('markdown-it-task-lists'));
    }
    
    public formatMessage(text: string): string {
        return this.md.render(text);
    }
}

// Component for displaying formatted messages
const FormattedMessage: React.FC<{content: string}> = ({ content }) => {
    const formatter = new MessageFormatter();
    const formattedContent = formatter.formatMessage(content);
    
    return (
        <View style={styles.messageContainer}>
            <RenderHtml
                contentWidth={width}
                source={{ html: formattedContent }}
                tagsStyles={markdownStyles}
            />
        </View>
    );
};

const markdownStyles = StyleSheet.create({
    p: {
        fontSize: 16,
        lineHeight: 24,
    },
    code: {
        fontFamily: 'monospace',
        backgroundColor: '#f5f5f5',
        padding: 4,
    },
    blockquote: {
        borderLeftWidth: 4,
        borderLeftColor: '#ddd',
        paddingLeft: 16,
        marginLeft: 0,
    }
});
```

Now let's create an input component that provides formatting shortcuts:

```typescript
const MessageInput: React.FC<{onSend: (text: string) => void}> = ({ onSend }) => {
    const [text, setText] = useState('');
    
    const formatSelection = (marker: string) => {
        const input = inputRef.current;
        if (input) {
            const { selectionStart, selectionEnd } = input;
            const selectedText = text.slice(selectionStart, selectionEnd);
            const newText = text.slice(0, selectionStart) +
                          `${marker}${selectedText}${marker}` +
                          text.slice(selectionEnd);
            setText(newText);
        }
    };
    
    return (
        <View style={styles.inputContainer}>
            <View style={styles.formatBar}>
                <TouchableOpacity onPress={() => formatSelection('**')}>
                    <FontAwesome name="bold" size={20} />
                </TouchableOpacity>
                <TouchableOpacity onPress={() => formatSelection('*')}>
                    <FontAwesome name="italic" size={20} />
                </TouchableOpacity>
                <TouchableOpacity onPress={() => formatSelection('`')}>
                    <FontAwesome name="code" size={20} />
                </TouchableOpacity>
            </View>
            <TextInput
                ref={inputRef}
                value={text}
                onChangeText={setText}
                multiline
                style={styles.input}
            />
            <TouchableOpacity onPress={() => onSend(text)}>
                <FontAwesome name="send" size={24} />
            </TouchableOpacity>
        </View>
    );
};
```

## Part 2: File Attachments and Image Sharing

Let's implement a robust file handling system that supports multiple file types and provides progress feedback:

```typescript
import * as DocumentPicker from 'expo-document-picker';
import * as ImagePicker from 'expo-image-picker';
import * as FileSystem from 'expo-file-system';

class FileHandler {
    // Maximum file size in bytes (10MB)
    private static MAX_FILE_SIZE = 10 * 1024 * 1024;
    
    // Supported file types
    private static SUPPORTED_TYPES = {
        images: ['image/jpeg', 'image/png', 'image/gif'],
        documents: ['application/pdf', 'text/plain'],
    };
    
    public async pickFile(): Promise<FileInfo | null> {
        try {
            const result = await DocumentPicker.getDocumentAsync({
                type: [...this.SUPPORTED_TYPES.images, 
                       ...this.SUPPORTED_TYPES.documents],
                copyToCacheDirectory: true
            });
            
            if (result.canceled) {
                return null;
            }
            
            const file = result.assets[0];
            
            // Validate file size
            if (file.size > FileHandler.MAX_FILE_SIZE) {
                throw new Error('File size exceeds limit');
            }
            
            return this.prepareFileForUpload(file);
            
        } catch (error) {
            console.error('Error picking file:', error);
            return null;
        }
    }
    
    private async prepareFileForUpload(file: any): Promise<FileInfo> {
        const fileInfo: FileInfo = {
            uri: file.uri,
            name: file.name,
            type: file.mimeType,
            size: file.size,
            thumbnail: null
        };
        
        // Generate thumbnail for images
        if (this.SUPPORTED_TYPES.images.includes(file.mimeType)) {
            fileInfo.thumbnail = await this.generateThumbnail(file.uri);
        }
        
        return fileInfo;
    }
    
    private async generateThumbnail(uri: string): Promise<string> {
        // Implement thumbnail generation logic
        // This could use Image manipulation libraries
        // Return thumbnail URI
    }
}

// Component for displaying file attachments
const FileAttachment: React.FC<{file: FileInfo}> = ({ file }) => {
    const [downloadProgress, setDownloadProgress] = useState(0);
    
    const handleDownload = async () => {
        const callback = (progress: number) => {
            setDownloadProgress(progress);
        };
        
        await FileSystem.downloadAsync(
            file.uri,
            FileSystem.documentDirectory + file.name,
            {
                md5: true,
                callback
            }
        );
    };
    
    return (
        <View style={styles.attachmentContainer}>
            {file.thumbnail ? (
                <Image source={{ uri: file.thumbnail }} 
                       style={styles.thumbnail} />
            ) : (
                <FileIcon type={file.type} />
            )}
            <Text>{file.name}</Text>
            <TouchableOpacity onPress={handleDownload}>
                <FontAwesome name="download" size={20} />
            </TouchableOpacity>
            {downloadProgress > 0 && (
                <ProgressBar progress={downloadProgress} />
            )}
        </View>
    );
};
```

## Part 3: Voice Input Integration

Let's add voice input support using the device's speech recognition capabilities:

```typescript
import * as Speech from 'expo-speech-recognition';

class VoiceInputHandler {
    private isListening: boolean = false;
    
    constructor() {
        this.requestPermissions();
    }
    
    private async requestPermissions() {
        const { status } = await Speech.requestPermissionsAsync();
        if (status !== 'granted') {
            throw new Error('Voice input permission denied');
        }
    }
    
    public async startListening(onResult: (text: string) => void) {
        if (this.isListening) return;
        
        this.isListening = true;
        
        try {
            await Speech.startListeningAsync({
                partialResults: true,
                onResults: (results) => {
                    const text = results.value[0];
                    onResult(text);
                }
            });
        } catch (error) {
            console.error('Error starting voice input:', error);
            this.isListening = false;
        }
    }
    
    public async stopListening() {
        if (!this.isListening) return;
        
        try {
            await Speech.stopListeningAsync();
        } finally {
            this.isListening = false;
        }
    }
}

// Voice input button component
const VoiceInputButton: React.FC<{onText: (text: string) => void}> = 
    ({ onText }) => {
    const [isRecording, setIsRecording] = useState(false);
    const voiceHandler = new VoiceInputHandler();
    
    const toggleRecording = async () => {
        if (isRecording) {
            await voiceHandler.stopListening();
        } else {
            await voiceHandler.startListening(onText);
        }
        setIsRecording(!isRecording);
    };
    
    return (
        <TouchableOpacity 
            onPress={toggleRecording}
            style={[
                styles.voiceButton,
                isRecording && styles.voiceButtonRecording
            ]}
        >
            <FontAwesome 
                name={isRecording ? "microphone-slash" : "microphone"} 
                size={24} 
            />
        </TouchableOpacity>
    );
};
```

## Part 4: Chat Export Functionality

Let's implement a system to export chat history in different formats:

```typescript
class ChatExporter {
    constructor(private messages: Message[]) {}
    
    public async exportToFormat(format: 'txt' | 'json' | 'html'): Promise<string> {
        switch (format) {
            case 'txt':
                return this.exportToText();
            case 'json':
                return this.exportToJson();
            case 'html':
                return this.exportToHtml();
            default:
                throw new Error('Unsupported format');
        }
    }
    
    private exportToText(): string {
        return this.messages.map(msg => {
            const timestamp = new Date(msg.timestamp)
                .toLocaleString();
            return `[${timestamp}] ${msg.sender}: ${msg.content}`;
        }).join('\n');
    }
    
    private exportToJson(): string {
        return JSON.stringify(this.messages, null, 2);
    }
    
    private exportToHtml(): string {
        // Create a styled HTML document with messages
        const template = `
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    .message { margin: 10px 0; }
                    .timestamp { color: #666; }
                    .sender { font-weight: bold; }
                </style>
            </head>
            <body>
                ${this.messages.map(msg => `
                    <div class="message">
                        <span class="timestamp">
                            ${new Date(msg.timestamp).toLocaleString()}
                        </span>
                        <span class="sender">${msg.sender}:</span>
                        <div class="content">${msg.content}</div>
                    </div>
                `).join('')}
            </body>
            </html>
        `;
        
        return template;
    }
}

// Export button component
const ExportButton: React.FC<{messages: Message[]}> = ({ messages }) => {
    const [showOptions, setShowOptions] = useState(false);
    
    const handleExport = async (format: 'txt' | 'json' | 'html') => {
        const exporter = new ChatExporter(messages);
        const content = await exporter.exportToFormat(format);
        
        // Save file
        const filename = `chat-export-${Date.now()}.${format}`;
        await FileSystem.writeAsStringAsync(
            FileSystem.documentDirectory + filename,
            content
        );
        
        // Share file
        await Sharing.shareAsync(
            FileSystem.documentDirectory + filename
        );
    };
    
    return (
        <View>
            <TouchableOpacity onPress={() => setShowOptions(true)}>
                <FontAwesome name="download" size={24} />
            </TouchableOpacity>
            
            <Modal visible={showOptions} animationType="slide">
                <View style={styles.exportOptions}>
                    <TouchableOpacity onPress={() => handleExport('txt')}>
                        <Text>Export as Text</Text>
                    </TouchableOpacity>
                    <TouchableOpacity onPress={() => handleExport('json')}>
                        <Text>Export as JSON</Text>
                    </TouchableOpacity>
                    <TouchableOpacity onPress={() => handleExport('html')}>
                        <Text>Export as HTML</Text>
                    </TouchableOpacity>
                </View>
            </Modal>
        </View>
    );
};
```

## Part 5: Message Reactions and Ratings

Finally, let's implement a reaction system that allows users to express their response to messages:

```typescript
interface Reaction {
    emoji: string;
    count: number;
    users: string[];
}

interface Message {
    // ... other message properties
    reactions: Record<string, Reaction>;
}

class ReactionHandler {
    private static AVAILABLE_REACTIONS = ['👍', '❤️', '😊', '🎉', '🤔', '👎'];
    
    public static toggleReaction(
        message: Message,
        emoji: string,
        userId: string
    ): Message {
        if (!message.reactions[emoji]) {
            message.reactions[emoji] = {
                emoji,
                count: 0,
                users: []
            };
        }
        
        const reaction = message.reactions[emoji];
        
        if (reaction.users.includes(userId)) {
            // Remove reaction
            reaction.users = reaction.users.filter(id => id !== userId);
            reaction.count--;
            
            if (reaction.count === 0) {
                delete message.reactions[emoji];
            }
        } else {
            // Add reaction
            reaction.users.push(userId);
            reaction.count++;
        }
        
        return { ...message };
    }
}

// Reaction picker component
const ReactionPicker: React.FC<{
    message: Message,
    onReact: (emoji: string) => void
}> = ({ message, onReact }) => {
    const [showPicker, setShowPicker] = useState(false);
    
    return (
        <View>
            <TouchableOpacity onPress={() => setShowPicker(true)}>
                <FontAwesome name="smile-o" size={20} />
            </TouchableOpacity>
            
            <Modal
                visible={showPicker}
                transparent
                animationType="fade"
                onRequestClose={() => setShowPicker(false)}
            >
                <View style={styles.reactionPicker}>
                    {ReactionHandler.AVAILABLE_REACTIONS.map(emoji