// app/(tabs)/index.tsx
import React, { useState, useEffect } from 'react';
import {
  StyleSheet,
  View,
  TextInput,
  Text,
  TouchableOpacity,
  ActivityIndicator,
  SafeAreaView,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  Alert,
  Modal,
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import axios from 'axios';
import { FontAwesome } from '@expo/vector-icons';
import { useAppContext } from '@/app/context/AppContext';

interface Model {
  name: string;
  url: string;
}

const localhost = "http://192.168.50.120:3000";

export default function TabOneScreen() {
  const { modelName, setModelName } = useAppContext();

  const [text, setText] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [inputHeight, setInputHeight] = useState(120);
  const [showSettings, setShowSettings] = useState(false);
  
  const [request, setRequest] = useState('');
  
  // LLM Parameters
  const [maxLength, setMaxLength] = useState('2048');
  const [maxTokens, setMaxTokens] = useState('500');
  // const [modelName, setModelName] = useState('TinyLlama/TinyLlama-1.1B-Chat-v1.0');
  
  const [settingsLoading, setSettingsLoading] = useState(false);
  
  const [availableModels, setAvailableModels] = useState<Model[]>([]);
  const [modelSelectVisible, setModelSelectVisible] = useState(false);
  const [loadingModels, setLoadingModels] = useState(false);
  
  useEffect(() => {
    fetchModels();
  }, []);
  
  const fetchModels = async () => {
    setLoadingModels(true);
    try {
      const response = await axios.get(`${localhost}/models`, {
		  params: {
			text: 't'
		  }
      });
      setAvailableModels(response.data);
    } catch (err) {
      console.error('Error fetching models:', err);
      Alert.alert('Error', 'Failed to load available models');
    } finally {
      setLoadingModels(false);
    }
  };

  const handleModelSelect = (model: Model) => {
    setModelName(model.url);
    setModelSelectVisible(false);
  };

  const ModelSelector = () => (
    <Modal
      visible={modelSelectVisible}
      transparent
      animationType="slide"
      onRequestClose={() => setModelSelectVisible(false)}
    >
      <View style={styles.modalOverlay}>
        <View style={styles.modalContent}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Select Model</Text>
            <TouchableOpacity
              onPress={() => setModelSelectVisible(false)}
              style={styles.closeButton}
            >
              <FontAwesome name="times" size={20} color="#666" />
            </TouchableOpacity>
          </View>

          <ScrollView style={styles.modelList}>
            {availableModels.map((model, index) => (
              <TouchableOpacity
                key={index}
                style={[
                  styles.modelItem,
                  model.url === modelName && styles.modelItemSelected
                ]}
                onPress={() => handleModelSelect(model)}
              >
                <Text style={[
                  styles.modelName,
                  model.url === modelName && styles.modelNameSelected
                ]}>
                  {model.name}
                </Text>
                <Text style={styles.modelUrl}>{model.url}</Text>
              </TouchableOpacity>
            ))}
          </ScrollView>
        </View>
      </View>
    </Modal>
  );

  // Modify the settings section to include the model selector
  const renderModelField = () => (
    <View style={styles.settingRow}>
      <TouchableOpacity
        style={styles.modelSelector}
        onPress={() => setModelSelectVisible(true)}
      >
        <Text style={styles.modelSelectorText}>
          {availableModels.find(m => m.url === modelName)?.name || modelName}
        </Text>
        <FontAwesome name="chevron-down" size={14} color="#666" />
      </TouchableOpacity>
    </View>
  );

  const handleUpdateSettings = async () => {
    if (!maxLength || !maxTokens || !modelName.trim()) {
      setError('Please fill in all settings fields');
      return;
    }

    setSettingsLoading(true);
    setError('');

    try {
      const response = await axios.get(`${localhost}/settings`, {
		  params: {
			maxLength: parseInt(maxLength),
			maxTokens: parseInt(maxTokens),
			model: modelName.trim()
		  }
      });

      if (response.data.status === 'success') {
        Alert.alert('Success', 'Settings updated successfully');
        setShowSettings(false);
      } else {
        setError('Failed to update settings');
      }
    } catch (err) {
      setError(err.message || 'Failed to update settings');
    } finally {
      setSettingsLoading(false);
    }
  };

  // Modified handleSubmit function in TabOneScreen to include model in history
	const handleSubmit = async () => {
	  if (!text.trim()) {
		setError('Please enter some text');
		return;
	  }

	  setLoading(true);
	  setError('');
	  setResponse('');
	  
	  prompt = text;
	  
	  setText('');

	  try {
		const result = await axios.get(`${localhost}/ask`, {
		  params: {
			text: prompt,
		  }
		});
		
		console.log(JSON.stringify(result.data, null, 2));
		
		setResponse(result.data.response);
		setRequest(prompt);
		
		await saveToHistory(prompt, result.data.response, modelName); // Include model name
	  } catch (err) {
		setError(err.message || 'An error occurred');
	  } finally {
		setLoading(false);
	  }
	};
  
	const saveToHistory = async (input: string, response: string, model: string) => {
	  try {
		const newItem = {
		  id: Date.now().toString(),
		  input,
		  response,
		  timestamp: new Date().toLocaleString(),
		  model, // Save the model information
      isRag: false,
		};
		
		const existingHistory = await AsyncStorage.getItem('feedbackHistory');
		const history = existingHistory ? JSON.parse(existingHistory) : [];
		const updatedHistory = [newItem, ...history];
		
		await AsyncStorage.setItem('feedbackHistory', JSON.stringify(updatedHistory));
	  } catch (error) {
		console.error('Error saving to history:', error);
	  }
	};

  return (
    <SafeAreaView style={styles.container}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.keyboardAvoid}
      >
        <ScrollView 
          style={styles.content}
          contentContainerStyle={styles.contentContainer}
          keyboardShouldPersistTaps="handled"
        >
          <View style={styles.headerContainer}>
            <Text style={styles.title}>Ask Me Anything</Text>
            <TouchableOpacity
              style={styles.settingsButton}
              onPress={() => setShowSettings(!showSettings)}
            >
              <FontAwesome 
                name={showSettings ? "chevron-up" : "chevron-down"} 
                size={16} 
                color="#007AFF" 
              />
              <Text style={styles.settingsButtonText}>
                {showSettings ? "Hide Settings" : "Show Settings"}
              </Text>
            </TouchableOpacity>
          </View>

          {showSettings && (
			<View style={styles.settingsContainer}>
			  <View style={styles.settingRow}>
				<Text style={styles.settingLabel}>Max Length:</Text>
				<TextInput
				  style={styles.settingInput}
				  value={maxLength}
				  onChangeText={setMaxLength}
				  keyboardType="numeric"
				  placeholder="2048"
				/>
			  </View>

			  <View style={styles.settingRow}>
				<Text style={styles.settingLabel}>Max Tokens:</Text>
				<TextInput
				  style={styles.settingInput}
				  value={maxTokens}
				  onChangeText={setMaxTokens}
				  keyboardType="numeric"
				  placeholder="500"
				/>
			  </View>

			  <View style={styles.settingRow}>
				<Text style={styles.settingLabel}>Model Name:</Text>
				{renderModelField()}
				<ModelSelector />
			  </View>

			  <TouchableOpacity
				style={[styles.updateButton, settingsLoading && styles.updateButtonDisabled]}
				onPress={handleUpdateSettings}
				disabled={settingsLoading}
			  >
				{settingsLoading ? (
				  <ActivityIndicator color="#fff" />
				) : (
				  <>
					<FontAwesome name="refresh" size={16} color="#fff" style={styles.updateIcon} />
					<Text style={styles.updateButtonText}>Update Settings</Text>
				  </>
				)}
			  </TouchableOpacity>
			</View>
		  )}
          
          <TextInput
            style={[styles.input, { height: Math.max(120, inputHeight) }]}
            value={text}
            onChangeText={setText}
            placeholder="Enter your text here..."
            multiline
            onContentSizeChange={(event) => {
              setInputHeight(event.nativeEvent.contentSize.height);
            }}
            placeholderTextColor="#666"
          />

          <TouchableOpacity 
            style={styles.button}
            onPress={handleSubmit}
            disabled={loading}
          >
            {loading ? (
              <ActivityIndicator color="#fff" />
            ) : (
              <Text style={styles.buttonText}>Tell Me</Text>
            )}
          </TouchableOpacity>

          {error ? (
            <Text style={styles.errorText}>{error}</Text>
          ) : null}

          {response ? (
            <View style={styles.responseContainer}>
              <Text style={styles.responseTitle}>Prompt:</Text> 
              <Text style={styles.requestText}>{request}</Text>
              <Text style={styles.responseTitle}>Answer:</Text>
              <ScrollView style={styles.responseScroll}>
                <Text style={styles.responseText}>{response}</Text>
              </ScrollView>
            </View>
          ) : null}
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  keyboardAvoid: {
    flex: 1,
  },
  content: {
    flex: 1,
  },
  contentContainer: {
    padding: 20,
    flexGrow: 1,
  },
  headerContainer: {
    flexDirection: 'column',
    alignItems: 'center',
    marginBottom: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    color: '#333',
    marginBottom: 10,
  },
  settingsButton: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 8,
  },
  settingsButtonText: {
    color: '#007AFF',
    marginLeft: 5,
    fontSize: 14,
  },
  settingsContainer: {
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 15,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: '#ddd',
  },
  settingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  settingLabel: {
    flex: 1,
    fontSize: 14,
    color: '#333',
    fontWeight: '500',
  },
  settingInput: {
    flex: 2,
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 6,
    padding: 8,
    fontSize: 14,
    backgroundColor: '#f9f9f9',
  },
  input: {
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 15,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: '#ddd',
    minHeight: 120,
    textAlignVertical: 'top',
    fontSize: 16,
  },
  button: {
    backgroundColor: '#007AFF',
    padding: 15,
    borderRadius: 8,
    alignItems: 'center',
    marginBottom: 20,
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  errorText: {
    color: '#ff3b30',
    marginBottom: 10,
    textAlign: 'center',
  },
  responseContainer: {
    backgroundColor: '#fff',
    padding: 15,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#ddd',
    maxHeight: 300,
  },
  responseScroll: {
    maxHeight: 250,
  },
  responseTitle: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 10,
    color: '#333',
  },
  responseText: {
    fontSize: 16,
    color: '#333',
    lineHeight: 24,
  },
  requestText: {
    fontSize: 16,
    color: '#333',
    lineHeight: 24,
    marginBottom: 10,
  },
  updateButton: {
    backgroundColor: '#28a745',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 12,
    borderRadius: 6,
    marginTop: 10,
  },
  updateButtonDisabled: {
    opacity: 0.7,
  },
  updateButtonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
  updateIcon: {
    marginRight: 8,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'flex-end',
  },
  modalContent: {
    backgroundColor: '#fff',
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    maxHeight: '80%',
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
  },
  closeButton: {
    padding: 5,
  },
  modelList: {
    padding: 15,
  },
  modelItem: {
    padding: 15,
    borderRadius: 8,
    marginBottom: 10,
    backgroundColor: '#f5f5f5',
  },
  modelItemSelected: {
    backgroundColor: '#e3effd',
    borderColor: '#007AFF',
    borderWidth: 1,
  },
  modelName: {
    fontSize: 16,
    fontWeight: '500',
    color: '#333',
    marginBottom: 4,
  },
  modelNameSelected: {
    color: '#007AFF',
  },
  modelUrl: {
    fontSize: 14,
    color: '#666',
  },
  modelSelector: {
    flex: 2,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#f9f9f9',
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 6,
    padding: 8,
    paddingHorizontal: 12,
  },
  modelSelectorText: {
    fontSize: 14,
    color: '#333',
    flex: 1,
    marginRight: 8,
  },
});
