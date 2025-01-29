// app/(tabs)/chat.tsx
import React, { useState, useEffect, useRef } from 'react';
import {
  StyleSheet,
  View,
  Text,
  TextInput,
  TouchableOpacity,
  FlatList,
  ActivityIndicator,
  SafeAreaView,
  KeyboardAvoidingView,
  Platform,
  Keyboard,
  ScrollView,
  Pressable,
} from 'react-native';
import { FontAwesome } from '@expo/vector-icons';
import AsyncStorage from '@react-native-async-storage/async-storage';
import axios from 'axios';
import { useAppContext } from '@/app/context/AppContext';
import Modal from 'react-native-modal';

const localhost = "http://192.168.50.120:3000";

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

interface ChatSession {
  id: string;
  name: string;
  lastMessage: string;
  timestamp: string;
}

export default function ChatScreen() {
  const { modelName } = useAppContext();
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSession, setCurrentSession] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [loading, setLoading] = useState(false);
  const [showNewChat, setShowNewChat] = useState(false);
  const [newChatName, setNewChatName] = useState('');
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [sessionToDelete, setSessionToDelete] = useState<string | null>(null);
  const [showSidebar, setShowSidebar] = useState(Platform.OS === 'web');
  const flatListRef = useRef<FlatList>(null);
  const inputRef = useRef<TextInput>(null);

  useEffect(() => {
    loadSessions();
  }, []);

  const loadSessions = async () => {
    try {
      const savedSessions = await AsyncStorage.getItem('chatSessions');
      if (savedSessions) {
        setSessions(JSON.parse(savedSessions));
      }
    } catch (error) {
      console.error('Error loading sessions:', error);
    }
  };

  const startNewChat = async () => {
    if (!newChatName.trim()) {
      return;
    }

    setLoading(true);
    try {
      const sessionId = Date.now().toString();
      const response = await axios.post(`${localhost}/chat/start`, {
        sessionId,
        initialMessage: '',
      });

      const newSession: ChatSession = {
        id: sessionId,
        name: newChatName.trim(),
        lastMessage: '',
        timestamp: new Date().toISOString(),
      };

      const updatedSessions = [newSession, ...sessions];
      setSessions(updatedSessions);
      await AsyncStorage.setItem('chatSessions', JSON.stringify(updatedSessions));
      
      setCurrentSession(newSession.id);
      setMessages([]);
      setNewChatName('');
      setShowNewChat(false);
      if (Platform.OS !== 'web') {
        setShowSidebar(false);
      }
    } catch (error) {
      console.error('Error starting new chat:', error);
    } finally {
      setLoading(false);
    }
  };

  const sendMessage = async () => {
    if (!inputText.trim() || !currentSession) return;

    const userMessage = inputText.trim();
    setInputText('');
    Keyboard.dismiss();

    const newMessage: Message = {
      role: 'user',
      content: userMessage,
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, newMessage]);

    setLoading(true);
    try {
      const response = await axios.post(`${localhost}/chat/message`, {
        sessionId: currentSession,
        message: userMessage
      });

      setMessages(response.data.messages);
      
      // Update session last message
      const updatedSessions = sessions.map(session => 
        session.id === currentSession 
          ? { ...session, lastMessage: userMessage, timestamp: new Date().toISOString() }
          : session
      );
      setSessions(updatedSessions);
      await AsyncStorage.setItem('chatSessions', JSON.stringify(updatedSessions));
      
      // Scroll to bottom
      setTimeout(() => {
        flatListRef.current?.scrollToEnd();
      }, 100);
    } catch (error) {
      console.error('Error sending message:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadChatHistory = async (sessionId: string) => {
    setLoading(true);
    try {
      const response = await axios.get(`${localhost}/chat/history/${sessionId}`);
      setMessages(response.data.messages || []);
      setCurrentSession(sessionId);
      if (Platform.OS !== 'web') {
        setShowSidebar(false);
      }
    } catch (error) {
      console.error('Error loading chat history:', error);
    } finally {
      setLoading(false);
    }
  };

  const deleteSession = async () => {
    if (!sessionToDelete) return;

    try {
      await axios.delete(`${localhost}/chat/session/${sessionToDelete}`);
      const updatedSessions = sessions.filter(session => session.id !== sessionToDelete);
      setSessions(updatedSessions);
      await AsyncStorage.setItem('chatSessions', JSON.stringify(updatedSessions));
      
      if (currentSession === sessionToDelete) {
        setCurrentSession(null);
        setMessages([]);
      }
    } catch (error) {
      console.error('Error deleting session:', error);
    } finally {
      setShowDeleteConfirm(false);
      setSessionToDelete(null);
    }
  };

  const renderMessage = ({ item }: { item: Message }) => (
    <View style={[
      styles.messageContainer,
      item.role === 'user' ? styles.userMessage : styles.assistantMessage,
    ]}>
      <Text style={[
        styles.messageText,
        item.role === 'assistant' && styles.assistantMessageText,
      ]} numberOfLines={0}>
        {item.content}
      </Text>
      <Text style={[
        styles.timestamp,
        item.role === 'assistant' && styles.assistantTimestamp
      ]}>
        {new Date(item.timestamp).toLocaleTimeString()}
      </Text>
    </View>
  );

  const renderSidebar = () => (
    <View style={[styles.sidebar, !showSidebar && styles.sidebarHidden]}>
      <TouchableOpacity
        style={styles.newChatButton}
        onPress={() => {
          setNewChatName('');
          setShowNewChat(true);
        }}
      >
        <FontAwesome name="plus" size={16} color="#fff" />
        <Text style={styles.newChatButtonText}>New Chat</Text>
      </TouchableOpacity>

      <ScrollView style={styles.sessionsList}>
        {sessions.map(session => (
          <TouchableOpacity
            key={session.id}
            style={[
              styles.sessionItem,
              currentSession === session.id && styles.activeSession
            ]}
            onPress={() => loadChatHistory(session.id)}
          >
            <View style={styles.sessionInfo}>
              <Text style={styles.sessionName}>{session.name}</Text>
              <Text style={styles.sessionLastMessage} numberOfLines={3}>
                {session.lastMessage || 'No messages yet'}
              </Text>
            </View>
            <TouchableOpacity
              style={styles.deleteButton}
              onPress={() => {
                setSessionToDelete(session.id);
                setShowDeleteConfirm(true);
              }}
            >
              <FontAwesome name="trash" size={16} color="#FF3B30" />
            </TouchableOpacity>
          </TouchableOpacity>
        ))}
      </ScrollView>
    </View>
  );

  const addSpaces = (howMany)=>{
    let spaces = "&nbsp; ";
    for (let i=0;i<howMany;i++){
      spaces += "&nbsp; ";
    }
    return spaces;
  }

  const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  keyboardAvoid: {
    flex: 1,
  },
  mainContainer: {
    flex: 1,
    flexDirection: 'row',
  },
  menuButton: {
    position: 'absolute',
    top: 10,
    left: 10,
    zIndex: 100,
    padding: 10,
    backgroundColor: '#fff',
    borderRadius: 8,
  },
  sidebar: {
    minWidth: 200,
    backgroundColor: '#fff',
    borderRightWidth: 1,
    borderRightColor: '#ddd',
    ...Platform.select({
      web: {
        minWidth: 150,
      },
      default: {
        position: 'absolute',
        top: 0,
        bottom: 0,
        left: 0,
        zIndex: 99,
      },
    }),
  },
  sidebarHidden: {
    ...Platform.select({
      web: {
        display: 'flex',
      },
      default: {
        display: 'none',
      },
    }),
  },
  newChatButton: {
    backgroundColor: '#007AFF',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 10,
    margin: 5,
    borderRadius: 8,
  },
  newChatButtonText: {
    color: '#fff',
    marginLeft: 8,
    fontSize: 16,
    fontWeight: '600',
  },
  sessionsList: {
    flex: 1,
  },
  sessionItem: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  activeSession: {
    backgroundColor: '#f0f0f0',
  },
  sessionInfo: {
    flex: 1,
  },
  sessionName: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 4,
    color: '#333',
  },
  sessionLastMessage: {
    fontSize: 14,
    color: '#666',
  },
  deleteButton: {
    paddingLeft: 16,
  },
  chatContainer: {
    flex: 1,
    backgroundColor: '#fff',
    marginLeft: Platform.OS === 'web' ? 0 : showSidebar ? 180 : 0,
  },
  messagesList: {
    padding: 15,
    flexGrow: 1,
  },

  messageContainer: {
    marginVertical: 5,
    padding: 12,
    borderRadius: 12,
    maxWidth: '80%',
    flexShrink: 1,
  },

  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 10,
    borderTopWidth: 1,
    borderTopColor: '#eee',
    backgroundColor: '#fff',
  },
  input: {
    flex: 1,
    backgroundColor: '#f5f5f5',
    borderRadius: 8,
    paddingHorizontal: 5,
    paddingVertical: 5,
    marginRight: 10,
    fontSize: 16,
    maxHeight: 200,
    width: "80%",
  },
  sendButton: {
    backgroundColor: '#007AFF',
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
  },
  sendButtonDisabled: {
    opacity: 0.5,
  },
  
  welcomeContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
    maxWidth: '100%',
  },

  welcomeText: {
    fontSize: 20,
    fontWeight: '600',
    color: '#333',
    marginBottom: 20,
    textAlign: 'center',
    flexWrap: 'wrap',
    flexShrink: 1,
    maxWidth: '100%',
  },
  welcomeButton: {
    backgroundColor: '#007AFF',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 15,
    borderRadius: 8,
    minWidth: 150,
  },
  welcomeButtonText: {
    color: '#fff',
    marginLeft: 8,
    fontSize: 16,
    fontWeight: '600',
  },
  modal: {
    margin: 0,
    justifyContent: 'center',
    ...Platform.select({
      web: {
        alignItems: 'center',
      },
    }),
  },
  modalContent: {
    backgroundColor: '#fff',
    borderRadius: 12,
    ...Platform.select({
      web: {
        width: '100%',
        maxWidth: 300,
      },
      default: {
        marginHorizontal: 20,
      },
    }),
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
    padding: 15,
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
  },
  closeButton: {
    padding: 5,
  },
  modalBody: {
    padding: 15,
  },
  modalLabel: {
    fontSize: 16,
    color: '#333',
    marginBottom: 10,
  },
  modalInput: {
    backgroundColor: '#f5f5f5',
    borderRadius: 8,
    padding: 12,
    fontSize: 16,
    marginBottom: 10,
  },
  modalMessage: {
    fontSize: 16,
    color: '#333',
    lineHeight: 22,
  },
  modalFooter: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    padding: 15,
    borderTopWidth: 1,
    borderTopColor: '#eee',
  },
  modalButton: {
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 8,
    minWidth: 80,
    alignItems: 'center',
    marginLeft: 10,
  },
  modalButtonCancel: {
    backgroundColor: '#e5e5ea',
  },
  modalButtonCreate: {
    backgroundColor: '#007AFF',
  },
  modalButtonDelete: {
    backgroundColor: '#FF3B30',
  },
  modalButtonText: {
    fontSize: 16,
    fontWeight: '500',
    color: '#000',
  },
  modalButtonCreateText: {
    color: '#fff',
  },
  modalButtonDeleteText: {
    color: '#fff',
  },
  // Responsive styles for different screen sizes
  ...Platform.select({
    web: {
      mainContainer: {
        minWidth: 375,
        maxWidth: 1200,
        alignSelf: 'center',
        flex: 1,
        flexDirection: 'row',
      },
      sidebar: {
        maxWidth: 180,
        minWidth: 100,
      },
      chatContainer: {
        flex: 1,
        borderLeftWidth: 1,
        borderLeftColor: '#ddd',
      },
      inputContainer: {
        width: "100%",
        maxWidth: 1200,
        flexDirection: 'row',
        alignItems: 'center',
        alignSelf: 'center',
        padding: 10,
        borderTopWidth: 1,
        borderTopColor: '#eee',
        backgroundColor: '#fff',
      },
    },
    ios: {
      mainContainer: {
        minWidth: 375,
        maxWidth: 1200,
        alignSelf: 'center',
        flex: 1,
        flexDirection: 'row',
      },
      sidebar: {
        maxWidth: 150,
        minWidth: 100,
      },
      chatContainer: {
        flex: 1,
        borderLeftWidth: 1,
        borderLeftColor: '#ddd',
      },
      inputContainer: {
        width: "100%",
        maxWidth: 1200,
        flexDirection: 'row',
        alignItems: 'center',
        alignSelf: 'center',
        padding: 10,
        borderTopWidth: 1,
        borderTopColor: '#eee',
        backgroundColor: '#fff',
      },
    },
    default: {
      mainContainer: {
        flex: 1,
      },
      chatContainer: {
        flex: 1,
      },
    },
  }),
  
  assistantMessage: {
    alignSelf: 'flex-start',
    backgroundColor: '#f0f0f0',
    maxWidth: '80%',
  },
  
  messageText: {
    fontSize: 16,
    color: '#fff',
    lineHeight: 22,
    flexWrap: 'wrap',
    flexShrink: 1,
    ...Platform.select({
      web: {
        wordBreak: 'break-word',
      },
      default: {
        // React Native specific
      }
    })
  },
  
  assistantMessageText: {
    color: '#333',
    flexWrap: 'wrap',
    flexShrink: 1,
  },
  
  timestamp: {
    fontSize: 12,
    color: 'rgba(255, 255, 255, 0.7)',
    marginTop: 4,
    alignSelf: 'flex-end',
  },
  
  assistantTimestamp: {
    color: 'rgba(0, 0, 0, 0.5)',
    alignSelf: 'flex-end',
  },

  userMessage: {
    alignSelf: 'flex-end',
    backgroundColor: '#007AFF',
    maxWidth: '80%',
  },
});

  return (
    <SafeAreaView style={styles.container}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.keyboardAvoid}
      >
        <View style={styles.mainContainer}>
          {Platform.OS !== 'web' && (
            <TouchableOpacity
              style={styles.menuButton}
              onPress={() => setShowSidebar(!showSidebar)}
            >
              <FontAwesome name={showSidebar ? "times" : "bars"} size={24} color="#007AFF" />
            </TouchableOpacity>
          )}

          {renderSidebar()}

          <View style={styles.chatContainer}>
            {currentSession ? (
              <>
                {(messages.length > 1) && <FlatList
                  ref={flatListRef}
                  data={messages}
                  renderItem={renderMessage}
                  keyExtractor={(item, index) => `${item.timestamp}-${index}`}
                  contentContainerStyle={styles.messagesList}
                  onContentSizeChange={() => flatListRef.current?.scrollToEnd()}
                />}

                {(messages.length === 0) && <View style={[styles.welcomeContainer]}>
                  <Text style={styles.welcomeText} numberOfLines={0}>
                    What would you like to discuss?
                    &nbsp; &nbsp; &nbsp; &nbsp; 
                    &nbsp; &nbsp; &nbsp; &nbsp; 
                    &nbsp; &nbsp; &nbsp; &nbsp; 
                    &nbsp; &nbsp; &nbsp; &nbsp; 
                    &nbsp; &nbsp; &nbsp; &nbsp;  
                    &nbsp; &nbsp; &nbsp; &nbsp;  
                    &nbsp; &nbsp; &nbsp; &nbsp;  
                    &nbsp; &nbsp; &nbsp; &nbsp;  
                    &nbsp; &nbsp; &nbsp; &nbsp;  
                    &nbsp; &nbsp; &nbsp; &nbsp;  
                    &nbsp; &nbsp; &nbsp; &nbsp;  
                    &nbsp; &nbsp; &nbsp; &nbsp;  
                    &nbsp; &nbsp; &nbsp; &nbsp;  
                    &nbsp; &nbsp; &nbsp; &nbsp;  
                    &nbsp; &nbsp; &nbsp; &nbsp;  
                    &nbsp; &nbsp; &nbsp; &nbsp;  
                  </Text>
                </View>}
              </>
            ) : (
              <View style={[styles.welcomeContainer]}>
                <Text style={styles.welcomeText} numberOfLines={0}>
                Select a chat session or start a new chat. 
                &nbsp; &nbsp; &nbsp; &nbsp; 
                &nbsp; &nbsp; &nbsp; &nbsp; 
                &nbsp; &nbsp; &nbsp; &nbsp; 
                &nbsp; &nbsp; &nbsp; &nbsp; 
                &nbsp; &nbsp; &nbsp; &nbsp;  
                &nbsp; &nbsp; &nbsp; &nbsp;  
                &nbsp; &nbsp; &nbsp; &nbsp;  
                &nbsp; &nbsp; &nbsp; &nbsp;  
                &nbsp; &nbsp; &nbsp; &nbsp;  
                &nbsp; &nbsp; &nbsp; &nbsp;  
                &nbsp; &nbsp; &nbsp; &nbsp;  
                &nbsp; &nbsp; &nbsp; &nbsp;  
                &nbsp; &nbsp; &nbsp; &nbsp;  
                &nbsp; &nbsp; &nbsp; &nbsp;  
                &nbsp; &nbsp; &nbsp; &nbsp;  
                &nbsp; &nbsp; &nbsp; &nbsp;  
                </Text>
              </View>
            )}
          </View>
        </View>

        <View style={styles.inputContainer}>
          <TextInput
            ref={inputRef}
            style={styles.input}
            value={inputText}
            onChangeText={setInputText}
            placeholder="Type your message..."
            multiline
            returnKeyType="send"
            onSubmitEditing={sendMessage}
            editable={!loading}
          />
          <TouchableOpacity
            style={[styles.sendButton, loading && styles.sendButtonDisabled]}
            onPress={sendMessage}
            disabled={loading}
          >
            {loading ? (
              <ActivityIndicator color="#fff" />
            ) : (
              <FontAwesome name="send" size={16} color="#fff" />
            )}
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>

      {/* New Chat Modal */}
      <Modal
        isVisible={showNewChat}
        onBackdropPress={() => setShowNewChat(false)}
        avoidKeyboard
        style={styles.modal}
      >
        <Pressable style={styles.modalContent} onPress={e => e.stopPropagation()}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>New Chat</Text>
            <TouchableOpacity
              onPress={() => setShowNewChat(false)}
              style={styles.closeButton}
            >
              <FontAwesome name="times" size={20} color="#666" />
            </TouchableOpacity>
          </View>

          <View style={styles.modalBody}>
            <Text style={styles.modalLabel}>Enter a name for your chat:</Text>
            <TextInput
              style={styles.modalInput}
              value={newChatName}
              onChangeText={setNewChatName}
              placeholder="Chat name..."
              autoFocus
            />
          </View>

          <View style={styles.modalFooter}>
            <TouchableOpacity
              style={[styles.modalButton, styles.modalButtonCancel]}
              onPress={() => {
                setShowNewChat(false);
                setNewChatName('');
              }}
            >
              <Text style={styles.modalButtonText}>Cancel</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.modalButton, styles.modalButtonCreate]}
              onPress={startNewChat}
            >
              <Text style={[styles.modalButtonText, styles.modalButtonCreateText]}>
                Create
              </Text>
            </TouchableOpacity>
          </View>
        </Pressable>
      </Modal>

      {/* Delete Confirmation Modal */}
      <Modal
        isVisible={showDeleteConfirm}
        onBackdropPress={() => setShowDeleteConfirm(false)}
        style={styles.modal}
      >
        <View style={styles.modalContent}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Delete Chat</Text>
          </View>

          <View style={styles.modalBody}>
            <Text style={styles.modalMessage}>
              Are you sure you want to delete this chat? This action cannot be undone.
            </Text>
          </View>

          <View style={styles.modalFooter}>
            <TouchableOpacity
              style={[styles.modalButton, styles.modalButtonCancel]}
              onPress={() => {
                setShowDeleteConfirm(false);
                setSessionToDelete(null);
              }}
            >
              <Text style={styles.modalButtonText}>Cancel</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.modalButton, styles.modalButtonDelete]}
              onPress={deleteSession}
            >
              <Text style={[styles.modalButtonText, styles.modalButtonDeleteText]}>
                Delete
              </Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
    </SafeAreaView>
  );
}

