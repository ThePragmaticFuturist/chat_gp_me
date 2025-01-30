# Understanding Client-Side Components in the LLM Chat Application

## Introduction

Imagine you're building a house. Just as a house needs a solid foundation, well-structured walls, and carefully planned rooms, our application needs a robust architecture. Today, we'll explore how different components work together to create a seamless user experience in our LLM chat application.

## The Foundation: Tab Navigation System

Think of our tab navigation system as the floor plan of our application. Just as a house has different rooms for different purposes, our app has different tabs for specific functionalities. Let's explore how this is implemented:

```typescript
// app/(tabs)/_layout.tsx
import { Tabs } from 'expo-router';
import { useColorScheme } from 'react-native';
import { FontAwesome } from '@expo/vector-icons';

export default function TabLayout() {
  // Get the device's color scheme (light/dark)
  const colorScheme = useColorScheme();

  return (
    <Tabs
      screenOptions={{
        // Configure common options for all tabs
        tabBarActiveTintColor: colorScheme === 'dark' ? '#fff' : '#2f95dc',
        tabBarStyle: {
          height: 60,
          paddingBottom: 5,
        },
      }}>
      
      {/* Home Tab - Quick Q&A Interface */}
      <Tabs.Screen
        name="index"
        options={{
          title: 'AMA',
          tabBarIcon: ({ color }) => (
            <FontAwesome name="comments" size={28} color={color} />
          ),
          headerTitle: 'Chat GP Me!',
        }}
      />

      {/* Chat Tab - Full Chat Experience */}
      <Tabs.Screen
        name="chat"
        options={{
          title: 'Chat',
          tabBarIcon: ({ color }) => (
            <FontAwesome name="comments" size={28} color={color} />
          ),
        }}
      />

      {/* History Tab - Past Conversations */}
      <Tabs.Screen
        name="two"
        options={{
          title: 'History',
          tabBarIcon: ({ color }) => (
            <FontAwesome name="history" size={28} color={color} />
          ),
        }}
      />

      {/* Documents Tab - RAG Interface */}
      <Tabs.Screen
        name="rag"
        options={{
          title: 'Docs',
          tabBarIcon: ({ color }) => (
            <FontAwesome name="file-text" size={28} color={color} />
          ),
        }}
      />
    </Tabs>
  );
}
```

Let's examine each tab's purpose and implementation:

### Home Tab (index.tsx)
This tab provides a simple, direct interaction with the LLM. Think of it as a quick consultation room where users can ask one-off questions. Here's how it works:

```typescript
// app/(tabs)/index.tsx
export default function HomeTab() {
  const [text, setText] = useState('');
  const [response, setResponse] = useState('');
  
  const handleSubmit = async () => {
    try {
      const result = await axios.get(`${localhost}/ask`, {
        params: { text: text.trim() }
      });
      setResponse(result.data.response);
      
      // Save interaction to history
      await saveToHistory(text, result.data.response, modelName);
    } catch (error) {
      Alert.alert('Error', error.message);
    }
  };
}
```

### Chat Tab (chat.tsx)
The chat tab is like a meeting room where users can have extended conversations. It maintains context and history within each chat session:

```typescript
// app/(tabs)/chat.tsx
export default function ChatTab() {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSession, setCurrentSession] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);

  const sendMessage = async (content: string) => {
    try {
      const response = await axios.post(`${localhost}/chat/message`, {
        sessionId: currentSession,
        message: content
      });
      
      // Update messages and session information
      setMessages(response.data.messages);
      updateSessionLastMessage(currentSession, content);
    } catch (error) {
      Alert.alert('Error', error.message);
    }
  };
}
```

### History Tab (two.tsx)
This tab serves as our archive room, where users can review past interactions:

```typescript
// app/(tabs)/two.tsx
export default function HistoryTab() {
  const [history, setHistory] = useState<FeedbackItem[]>([]);

  // Load history when component mounts
  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = async () => {
    try {
      const savedHistory = await AsyncStorage.getItem('feedbackHistory');
      if (savedHistory) {
        setHistory(JSON.parse(savedHistory));
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to load history');
    }
  };
}
```

## State Management: The Nervous System

Just as our nervous system coordinates different parts of our body, our state management system coordinates different parts of our application. We use React Context for this purpose:

```typescript
// app/context/AppContext.tsx
interface AppContextType {
  modelName: string;
  setModelName: (name: string) => void;
  logThis: (label: string, data: any) => void;
}

export const AppProvider: React.FC<AppProviderProps> = ({ children }) => {
  // Global state for model selection
  const [modelName, setModelName] = useState(defaultContext.modelName);

  // Utility function for debugging
  const logThis = (label: string, data: any) => {
    console.log(`[${label}]`, data);
  };

  return (
    <AppContext.Provider
      value={{
        modelName,
        setModelName,
        logThis,
      }}
    >
      {children}
    </AppContext.Provider>
  );
};
```

To use this global state in any component:

```typescript
function MyComponent() {
  const { modelName, setModelName } = useAppContext();
  
  // Now you can access and modify the global state
  const changeModel = (newModel: string) => {
    setModelName(newModel);
  };
}
```

## UI Components: The Building Blocks

Just as we use standard building materials to construct different parts of a house, we use reusable UI components to build our interface. Let's explore some key components:

### AlertModal: The Communication System

This component handles important notifications and confirmations:

```typescript
// components/AlertModal.tsx
interface AlertModalProps {
  visible: boolean;
  onClose: () => void;
  title: string;
  message: string;
  buttons: Button[];
}

const AlertModal: React.FC<AlertModalProps> = ({
  visible,
  onClose,
  title,
  message,
  buttons,
}) => {
  return (
    <Modal
      visible={visible}
      transparent
      animationType="fade"
      onRequestClose={onClose}
    >
      <View style={styles.overlay}>
        <View style={styles.content}>
          <Text style={styles.title}>{title}</Text>
          <Text style={styles.message}>{message}</Text>
          <View style={styles.buttonContainer}>
            {buttons.map((button, index) => (
              <TouchableOpacity
                key={index}
                style={[styles.button, getButtonStyle(button.style)]}
                onPress={button.onPress}
              >
                <Text style={styles.buttonText}>{button.text}</Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>
      </View>
    </Modal>
  );
};
```

### ThemedText and ThemedView: The Adaptive Interface

These components automatically adjust their appearance based on the device's theme:

```typescript
// components/ThemedText.tsx
export function ThemedText({
  style,
  lightColor,
  darkColor,
  type = 'default',
  ...rest
}: ThemedTextProps) {
  // Get the appropriate color based on theme
  const color = useThemeColor({ light: lightColor, dark: darkColor }, 'text');

  return (
    <Text
      style={[
        { color },
        getTypeStyle(type),
        style,
      ]}
      {...rest}
    />
  );
}

// components/ThemedView.tsx
export function ThemedView({ style, lightColor, darkColor, ...otherProps }: ThemedViewProps) {
  const backgroundColor = useThemeColor(
    { light: lightColor, dark: darkColor },
    'background'
  );

  return <View style={[{ backgroundColor }, style]} {...otherProps} />;
}
```

### Collapsible: The Expandable Sections

This component creates expandable sections for organizing content:

```typescript
// components/Collapsible.tsx
export function Collapsible({
  children,
  title
}: PropsWithChildren & { title: string }) {
  const [isOpen, setIsOpen] = useState(false);
  const theme = useColorScheme() ?? 'light';

  return (
    <ThemedView>
      <TouchableOpacity
        style={styles.heading}
        onPress={() => setIsOpen(!isOpen)}
      >
        <IconSymbol
          name="chevron.right"
          size={18}
          weight="medium"
          color={getIconColor(theme)}
          style={{
            transform: [{ rotate: isOpen ? '90deg' : '0deg' }]
          }}
        />
        <ThemedText type="defaultSemiBold">{title}</ThemedText>
      </TouchableOpacity>
      {isOpen && (
        <ThemedView style={styles.content}>
          {children}
        </ThemedView>
      )}
    </ThemedView>
  );
}
```

## Putting It All Together

Let's see how these components work together in a real scenario:

```typescript
function ExampleScreen() {
  const { modelName } = useAppContext();
  const [showAlert, setShowAlert] = useState(false);

  return (
    <ThemedView style={styles.container}>
      <Collapsible title="Settings">
        <ThemedText>Selected Model: {modelName}</ThemedText>
      </Collapsible>

      <AlertModal
        visible={showAlert}
        title="Confirmation"
        message="Are you sure?"
        buttons={[
          {
            text: "Cancel",
            style: "cancel",
            onPress: () => setShowAlert(false)
          },
          {
            text: "Confirm",
            onPress: () => handleConfirmation()
          }
        ]}
      />
    </ThemedView>
  );
}
```

## Practice Exercises

To better understand these components, try these exercises:

1. Create a new tab that combines features from different existing tabs
2. Add a new theme color and implement it across the application
3. Create a new reusable component that uses the theming system
4. Implement a new context provider for managing user preferences
5. Build a custom modal component using the existing components

## Common Challenges and Solutions

1. Theme Consistency
   - Always use ThemedText and ThemedView for consistent appearance
   - Create a central theme configuration file
   - Test in both light and dark modes

2. State Management
   - Keep context providers focused on specific concerns
   - Use local state for component-specific data
   - Consider performance implications of context updates

3. Navigation
   - Handle deep linking properly
   - Manage navigation state persistence
   - Handle navigation edge cases

## Conclusion

Understanding how these components work together is crucial for building a robust application. Remember:

1. The tab navigation system provides the core structure
2. Context management handles global state effectively
3. Reusable UI components ensure consistency
4. Proper component organization improves maintainability
5. Theme awareness creates a better user experience

Keep practicing with these components, and you'll become proficient at building complex, maintainable React Native applications! [Next: Vector Db](https://github.com/ThePragmaticFuturist/chat_gp_me/blob/main/webinars/AI%20Chat%20Course/5_vector-db-lesson.md)
