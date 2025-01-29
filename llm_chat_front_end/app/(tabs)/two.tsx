// app/(tabs)/two.tsx
import React, { useState, useEffect } from 'react';
import {
  StyleSheet,
  View,
  Text,
  FlatList,
  TouchableOpacity,
  SafeAreaView,
  Alert,
  ActivityIndicator,
} from 'react-native';
import { useFocusEffect } from 'expo-router';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { FontAwesome } from '@expo/vector-icons';
import AlertModal from '@/components/modal';

interface FeedbackItem {
  id: string;
  input: string;
  response: string;
  timestamp: string;
  model: string;
}

export default function TabTwoScreen() {
  const [history, setHistory] = useState<FeedbackItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [selectedItem, setSelectedItem] = useState<FeedbackItem | null>(null);
  const [showClearConfirm, setShowClearConfirm] = useState(false);

  // Load history when the component mounts and when the tab is focused
  useFocusEffect(
    React.useCallback(() => {
      loadHistory();
    }, [])
  );

  const loadHistory = async () => {
    try {
      const savedHistory = await AsyncStorage.getItem('feedbackHistory');
      if (savedHistory) {
        const parsedHistory = JSON.parse(savedHistory);
        setHistory(parsedHistory);
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to load history');
    } finally {
      setLoading(false);
    }
  };

  const clearHistory = () => {
    setShowClearConfirm(true);
  };

  const viewFullFeedback = (item: FeedbackItem) => {
    setSelectedItem(item);
    setIsModalVisible(true);
  };

  const renderItem = ({ item }: { item: FeedbackItem }) => (
    <TouchableOpacity 
      style={styles.historyItem}
      onPress={() => viewFullFeedback(item)}
    >
      <View style={styles.historyHeader}>
        <Text style={styles.timestamp}>{item.timestamp}</Text>
      </View>
      
      <View style={styles.modelContainer}>
        <Text style={styles.modelLabel}>Model:</Text>
        <Text style={styles.modelText}>
          {item.model || 'Unknown'}
        </Text>
      </View>

      <View style={styles.inputContainer}>
        <Text style={styles.label}>Input:</Text>
        <Text style={styles.text} numberOfLines={2}>
          {item.input}
        </Text>
      </View>

      <View style={styles.responseContainer}>
        <Text style={styles.label}>Response:</Text>
        <Text style={styles.text} numberOfLines={3}>
          {item.response}
        </Text>
      </View>

      <TouchableOpacity
        style={styles.expandButton}
        onPress={() => viewFullFeedback(item)}
      >
        <Text style={styles.expandButtonText}>View Full</Text>
      </TouchableOpacity>
    </TouchableOpacity>
  );

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#007AFF" />
      </View>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      {history.length > 0 ? (
        <>
          <TouchableOpacity
            style={styles.clearButton}
            onPress={clearHistory}
          >
            <FontAwesome name="trash" size={16} color="#FF3B30" />
            <Text style={styles.clearButtonText}>Clear History</Text>
          </TouchableOpacity>

          <FlatList
            data={history}
            renderItem={renderItem}
            keyExtractor={(item) => item.id}
            contentContainerStyle={styles.listContainer}
          />
        </>
      ) : (
        <View style={styles.emptyContainer}>
          <FontAwesome name="history" size={50} color="#999" />
          <Text style={styles.emptyText}>No feedback history yet</Text>
        </View>
      )}
      
      <AlertModal
        visible={isModalVisible}
        onClose={() => setIsModalVisible(false)}
        title="Full Feedback"
        message={selectedItem ? `Model: ${selectedItem.model || 'Unknown'}\n\nInput:\n${selectedItem.input}\n\nResponse:\n${selectedItem.response}` : ''}
        buttons={[
          {
            text: 'Close',
            style: 'cancel',
            onPress: () => setIsModalVisible(false)
          }
        ]}
      />

      <AlertModal
        visible={showClearConfirm}
        onClose={() => setShowClearConfirm(false)}
        title="Clear History"
        message="Are you sure you want to clear all history?"
        buttons={[
          {
            text: 'Cancel',
            style: 'cancel',
            onPress: () => setShowClearConfirm(false)
          },
          {
            text: 'Clear',
            style: 'destructive',
            onPress: async () => {
              try {
                await AsyncStorage.removeItem('feedbackHistory');
                setHistory([]);
                setShowClearConfirm(false);
              } catch (error) {
                console.error('Failed to clear history:', error);
                Alert.alert('Error', 'Failed to clear history');
              }
            }
          }
        ]}
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  listContainer: {
    padding: 15,
  },
  historyItem: {
    backgroundColor: '#fff',
    borderRadius: 10,
    padding: 15,
    marginBottom: 15,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 3,
    elevation: 3,
  },
  historyHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 10,
  },
  timestamp: {
    color: '#666',
    fontSize: 12,
  },
  inputContainer: {
    marginBottom: 10,
  },
  responseContainer: {
    marginBottom: 10,
  },
  label: {
    fontWeight: '600',
    fontSize: 14,
    color: '#333',
    marginBottom: 4,
  },
  text: {
    fontSize: 14,
    color: '#666',
  },
  expandButton: {
    alignSelf: 'flex-end',
  },
  expandButtonText: {
    color: '#007AFF',
    fontSize: 14,
  },
  clearButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 10,
    backgroundColor: '#fff',
    borderRadius: 8,
    margin: 15,
    marginBottom: 0,
  },
  clearButtonText: {
    color: '#FF3B30',
    marginLeft: 8,
    fontSize: 16,
    fontWeight: '500',
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  emptyText: {
    marginTop: 10,
    fontSize: 16,
    color: '#999',
  },
  modelContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
    backgroundColor: '#f5f5f5',
    padding: 8,
    borderRadius: 6,
  },
  modelLabel: {
    fontWeight: '600',
    fontSize: 14,
    color: '#333',
    marginRight: 8,
  },
  modelText: {
    fontSize: 14,
    color: '#666',
    flex: 1,
  },
});
