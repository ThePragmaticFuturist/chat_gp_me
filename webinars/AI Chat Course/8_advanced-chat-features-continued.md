### Continuing with Message Reactions and Ratings

Let's complete our reaction system implementation and add message ratings functionality. These features help users express their responses to messages in a more nuanced way.

```typescript
// Continuing from the ReactionPicker component
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

