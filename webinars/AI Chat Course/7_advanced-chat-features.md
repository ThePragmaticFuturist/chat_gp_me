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
        <View style={styles.reactionContainer}>
            <TouchableOpacity 
                onPress={() => setShowPicker(true)}
                style={styles.reactionButton}
            >
                <FontAwesome name="smile-o" size={20} />
            </TouchableOpacity>
            
            <Modal
                visible={showPicker}
                transparent
                animationType="fade"
                onRequestClose={() => setShowPicker(false)}
            >
                <TouchableOpacity 
                    style={styles.modalOverlay}
                    onPress={() => setShowPicker(false)}
                >
                    <View style={styles.reactionPicker}>
                        {ReactionHandler.AVAILABLE_REACTIONS.map(emoji => (
                            <TouchableOpacity
                                key={emoji}
                                style={styles.emojiButton}
                                onPress={() => {
                                    onReact(emoji);
                                    setShowPicker(false);
                                }}
                            >
                                <Text style={styles.emoji}>{emoji}</Text>
                            </TouchableOpacity>
                        ))}
                    </View>
                </TouchableOpacity>
            </Modal>
            
            {/* Display existing reactions */}
            <View style={styles.reactionList}>
                {Object.values(message.reactions).map(reaction => (
                    <View key={reaction.emoji} style={styles.reactionBubble}>
                        <Text style={styles.reactionEmoji}>
                            {reaction.emoji}
                        </Text>
                        <Text style={styles.reactionCount}>
                            {reaction.count}
                        </Text>
                    </View>
                ))}
            </View>
        </View>
    );
};

// Adding a Rating System for Messages
interface Rating {
    value: number;      // Rating value (1-5)
    userId: string;     // User who gave the rating
    timestamp: string;  // When the rating was given
    comment?: string;   // Optional comment
}

interface Message {
    // ... existing message properties
    ratings: Rating[];
    averageRating: number;
}

class RatingHandler {
    public static addRating(
        message: Message,
        rating: Rating
    ): Message {
        // Remove any existing rating by this user
        const filteredRatings = message.ratings.filter(
            r => r.userId !== rating.userId
        );
        
        // Add new rating
        const updatedRatings = [...filteredRatings, rating];
        
        // Calculate new average
        const average = updatedRatings.reduce(
            (sum, r) => sum + r.value, 0
        ) / updatedRatings.length;
        
        return {
            ...message,
            ratings: updatedRatings,
            averageRating: Number(average.toFixed(1))
        };
    }
}

// Rating component with star display
const MessageRating: React.FC<{
    message: Message,
    onRate: (rating: Rating) => void,
    currentUserId: string
}> = ({ message, onRate, currentUserId }) => {
    const [showRatingInput, setShowRatingInput] = useState(false);
    const [ratingValue, setRatingValue] = useState(0);
    const [comment, setComment] = useState('');
    
    const handleSubmitRating = () => {
        if (ratingValue === 0) return;
        
        const rating: Rating = {
            value: ratingValue,
            userId: currentUserId,
            timestamp: new Date().toISOString(),
            comment: comment.trim() || undefined
        };
        
        onRate(rating);
        setShowRatingInput(false);
        setRatingValue(0);
        setComment('');
    };
    
    return (
        <View style={styles.ratingContainer}>
            {/* Display average rating */}
            <View style={styles.averageRating}>
                <StarDisplay rating={message.averageRating} />
                <Text style={styles.ratingCount}>
                    ({message.ratings.length})
                </Text>
            </View>
            
            <TouchableOpacity
                onPress={() => setShowRatingInput(true)}
                style={styles.rateButton}
            >
                <Text>Rate this response</Text>
            </TouchableOpacity>
            
            <Modal
                visible={showRatingInput}
                transparent
                animationType="slide"
            >
                <View style={styles.ratingModal}>
                    <Text style={styles.ratingTitle}>
                        Rate this response
                    </Text>
                    
                    <StarInput
                        value={ratingValue}
                        onValueChange={setRatingValue}
                    />
                    
                    <TextInput
                        style={styles.commentInput}
                        value={comment}
                        onChangeText={setComment}
                        placeholder="Add a comment (optional)"
                        multiline
                    />
                    
                    <View style={styles.ratingButtons}>
                        <TouchableOpacity
                            style={styles.cancelButton}
                            onPress={() => setShowRatingInput(false)}
                        >
                            <Text>Cancel</Text>
                        </TouchableOpacity>
                        
                        <TouchableOpacity
                            style={[
                                styles.submitButton,
                                ratingValue === 0 && styles.submitDisabled
                            ]}
                            onPress={handleSubmitRating}
                            disabled={ratingValue === 0}
                        >
                            <Text>Submit Rating</Text>
                        </TouchableOpacity>
                    </View>
                </View>
            </Modal>
        </View>
    );
};

// Star display component for ratings
const StarDisplay: React.FC<{rating: number}> = ({ rating }) => {
    const fullStars = Math.floor(rating);
    const hasHalfStar = rating % 1 >= 0.5;
    
    return (
        <View style={styles.starContainer}>
            {[1, 2, 3, 4, 5].map(i => (
                <FontAwesome
                    key={i}
                    name={
                        i <= fullStars
                            ? 'star'
                            : i === fullStars + 1 && hasHalfStar
                            ? 'star-half-o'
                            : 'star-o'
                    }
                    size={16}
                    color="#FFD700"
                />
            ))}
        </View>
    );
};

// Interactive star input component
const StarInput: React.FC<{
    value: number,
    onValueChange: (value: number) => void
}> = ({ value, onValueChange }) => {
    return (
        <View style={styles.starInputContainer}>
            {[1, 2, 3, 4, 5].map(i => (
                <TouchableOpacity
                    key={i}
                    onPress={() => onValueChange(i)}
                    style={styles.starButton}
                >
                    <FontAwesome
                        name={i <= value ? 'star' : 'star-o'}
                        size={32}
                        color="#FFD700"
                    />
                </TouchableOpacity>
            ))}
        </View>
    );
};

// Styles for our components
const styles = StyleSheet.create({
    reactionContainer: {
        flexDirection: 'row',
        alignItems: 'center',
        marginTop: 8,
    },
    modalOverlay: {
        flex: 1,
        backgroundColor: 'rgba(0, 0, 0, 0.5)',
        justifyContent: 'center',
        alignItems: 'center',
    },
    reactionPicker: {
        backgroundColor: '#fff',
        borderRadius: 12,
        padding: 12,
        flexDirection: 'row',
        justifyContent: 'space-around',
        width: '80%',
        maxWidth: 400,
    },
    emojiButton: {
        padding: 8,
    },
    emoji: {
        fontSize: 24,
    },
    reactionList: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        marginLeft: 8,
    },
    reactionBubble: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: '#f0f0f0',
        borderRadius: 12,
        padding: 4,
        marginRight: 4,
        marginBottom: 4,
    },
    reactionEmoji: {
        fontSize: 16,
    },
    reactionCount: {
        fontSize: 12,
        marginLeft: 4,
        color: '#666',
    },
    ratingContainer: {
        marginTop: 12,
    },
    averageRating: {
        flexDirection: 'row',
        alignItems: 'center',
    },
    ratingCount: {
        marginLeft: 8,
        color: '#666',
        fontSize: 14,
    },
    rateButton: {
        marginTop: 8,
        padding: 8,
        backgroundColor: '#f0f0f0',
        borderRadius: 8,
        alignSelf: 'flex-start',
    },
    ratingModal: {
        backgroundColor: '#fff',
        borderRadius: 16,
        padding: 20,
        width: '90%',
        maxWidth: 400,
        alignSelf: 'center',
    },
    starContainer: {
        flexDirection: 'row',
        alignItems: 'center',
    },
    starInputContainer: {
        flexDirection: 'row',
        justifyContent: 'center',
        marginVertical: 20,
    },
    starButton: {
        padding: 4,
    },
    commentInput: {
        borderWidth: 1,
        borderColor: '#ddd',
        borderRadius: 8,
        padding: 12,
        marginTop: 16,
        minHeight: 80,
    },
    ratingButtons: {
        flexDirection: 'row',
        justifyContent: 'flex-end',
        marginTop: 16,
    },
    submitButton: {
        backgroundColor: '#007AFF',
        padding: 12,
        borderRadius: 8,
        marginLeft: 12,
    },
    submitDisabled: {
        opacity: 0.5,
    },
    cancelButton: {
        padding: 12,
    },
});

// Usage Example in a Message Component
const MessageComponent: React.FC<{
    message: Message,
    currentUserId: string,
    onReact: (messageId: string, emoji: string) => void,
    onRate: (messageId: string, rating: Rating) => void
}> = ({ message, currentUserId, onReact, onRate }) => {
    return (
        <View style={styles.messageContainer}>
            <Text style={styles.messageContent}>
                {message.content}
            </Text>
            
            <ReactionPicker
                message={message}
                onReact={(emoji) => onReact(message.id, emoji)}
            />
            
            <MessageRating
                message={message}
                currentUserId={currentUserId}
                onRate={(rating) => onRate(message.id, rating)}
            />
        </View>
    );
};
```

This implementation provides a complete system for message reactions and ratings, with the following features:

1. Quick emoji reactions with a popup picker
2. Visual display of reaction counts
3. Five-star rating system with half-star support
4. Optional comments with ratings
5. Average rating calculation and display
6. User-specific rating tracking
7. Smooth animations and intuitive UI
8. Proper modal handling for both reactions and ratings
9. Responsive design that works on both mobile and web

The components are designed to be reusable and maintainable, with clear separation of concerns between the data handling (ReactionHandler, RatingHandler) and the UI components (ReactionPicker, MessageRating).

For optimal performance, consider implementing the following optimizations:

1. Memoize components to prevent unnecessary re-renders
2. Batch rating and reaction updates
3. Implement optimistic updates for better UX
4. Cache reaction and rating data locally
5. Implement proper error handling and retry logic for network operations

Remember to handle edge cases such as:
- Network connectivity issues
- Concurrent reactions/ratings from multiple users
- Long comments in ratings
- Large numbers of reactions
- Rating changes and updates
- User permissions and restrictions

This implementation provides a solid foundation that you can build upon based on your specific needs and requirements.
