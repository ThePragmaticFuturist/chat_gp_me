// app/(tabs)/two.tsx
import React, { useState, useEffect } from 'react';
import {
  StyleSheet,
  View,
  Text,
  TextInput,
  TouchableOpacity,
  ScrollView,
  ActivityIndicator,
  SafeAreaView,
  Alert,
  Platform,
  KeyboardAvoidingView,
} from 'react-native';
import Modal from 'react-native-modal';
import AlertModal from '@/components/modal';
import * as DocumentPicker from 'expo-document-picker';
import { FontAwesome } from '@expo/vector-icons';
import AsyncStorage from '@react-native-async-storage/async-storage';
import axios from 'axios';

import { useAppContext } from '@/app/context/AppContext';

const localhost = "http://192.168.50.120:3000";

interface UploadedFile {
  id: string;
  name: string;
  timestamp: string;
  documentIds: string[];  // Array of Chroma document IDs
}

interface DocumentChunk {
  id: string;
  content: string;
  metadata: any;
}

export default function RagScreen() {
  const { modelName, setModelName, logThis } = useAppContext();
  
  const [text, setText] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [uploading, setUploading] = useState(false);
  const [inputHeight, setInputHeight] = useState(120);
  const [request, setRequest] = useState('');
  const [showClearConfirm, setShowClearConfirm] = useState(false);
  const [selectedFile, setSelectedFile] = useState<UploadedFile | null>(null);
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [documentChunks, setDocumentChunks] = useState<DocumentChunk[]>([]);
  const [loadingChunks, setLoadingChunks] = useState(false);

  useEffect(() => {
    loadUploadedFiles();
  }, []);

  // Add proper type checking for AsyncStorage data
  const loadUploadedFiles = async () => {
    try {
      const files = await AsyncStorage.getItem('uploadedFiles');
      if (files) {
        const parsedFiles = JSON.parse(files);
        if (Array.isArray(parsedFiles)) {
          setUploadedFiles(parsedFiles);
        }
      }
    } catch (error) {
      console.error('Error loading files:', error);
    }
  };

  const handleUpload = async () => {
    try {
      setUploading(true);
      const result = await DocumentPicker.getDocumentAsync({
        type: ['application/pdf', 'text/plain'],
        copyToCacheDirectory: true,
      });

      if (result.canceled) {
        return;
      }

      const file = result.assets[0];
      let formData = new FormData();

      // Handle base64 data from web platform
      if (file.uri.startsWith('data:')) {
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

      const uploadResponse = await axios.post(`${localhost}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (!uploadResponse.data.document_ids) {
        throw new Error('No document IDs received from server');
      }

      const newFile: UploadedFile = {
        id: Date.now().toString(),
        name: file.name,
        timestamp: new Date().toLocaleString(),
        documentIds: uploadResponse.data.document_ids,
      };

      const updatedFiles = [...uploadedFiles, newFile];
      setUploadedFiles(updatedFiles);
      await AsyncStorage.setItem('uploadedFiles', JSON.stringify(updatedFiles));

      Alert.alert('Success', 'File uploaded and processed successfully');
    } catch (error) {
      console.error('Upload error:', error);
      Alert.alert(
        'Error',
        error.response?.data?.error || error.message || 'Failed to upload file'
      );
    } finally {
      setUploading(false);
    }
  };

  const handleDocumentPress = async (file: UploadedFile) => {
    console.log("select", file);
    
    if (!file.documentIds || file.documentIds.length === 0) {
      Alert.alert('Error', 'No document chunks found for this file');
      return;
    }

    setSelectedFile(file);
    setIsModalVisible(true);
    setLoadingChunks(true);

    try {
      const response = await axios.post(`${localhost}/get_texts`, {
        ids: file.documentIds  // Send the array of Chroma document IDs
      });

      if (response.data.documents) {
        setDocumentChunks(response.data.documents);
      }
    } catch (error) {
      console.error('Error fetching document chunks:', error);
      Alert.alert('Error', 'Failed to fetch document chunks');
    } finally {
      setLoadingChunks(false);
    }
  };

  const handleDeleteChunk = async (chunkId: string) => {
    try {
      console.log("chunkId", chunkId);

      await axios.delete(`${localhost}/delete_chunk/${chunkId}`);
      
      // Update the chunks list
      setDocumentChunks(prevChunks => prevChunks.filter(chunk => chunk.id !== chunkId));
      
      // Update the file's documentIds
      if (selectedFile) {
        const updatedFile = {
          ...selectedFile,
          documentIds: selectedFile.documentIds?.filter(id => id !== chunkId)
        };
        
        setUploadedFiles(prevFiles => 
          prevFiles.map(file => 
            file.id === selectedFile.id ? updatedFile : file
          )
        );
        
        // Update AsyncStorage
        const updatedFiles = uploadedFiles.map(file => 
          file.id === selectedFile.id ? updatedFile : file
        );
        await AsyncStorage.setItem('uploadedFiles', JSON.stringify(updatedFiles));
        
        setSelectedFile(updatedFile);
      }
    } catch (error) {
      console.error('Error deleting chunk:', error);
      Alert.alert('Error', 'Failed to delete chunk');
    }
  };

  const handleDeleteDocument = async () => {
    if (!selectedFile?.documentIds || selectedFile.documentIds.length === 0) {
      Alert.alert('Error', 'No document IDs found for this file');
      return;
    }

    try {
      console.log('Deleting document with IDs:', selectedFile.documentIds);
      
      const response = await axios.delete(`${localhost}/delete_document`, {
        data: { 
          ids: selectedFile.documentIds
        },
        headers: {
          'Content-Type': 'application/json'
        }
      });

      console.log('Delete response:', response.data);

      // Update local state
      setUploadedFiles(prevFiles => 
        prevFiles.filter(file => file.id !== selectedFile.id)
      );
      
      // Update AsyncStorage
      const updatedFiles = uploadedFiles.filter(file => file.id !== selectedFile.id);
      await AsyncStorage.setItem('uploadedFiles', JSON.stringify(updatedFiles));
      
      setIsModalVisible(false);
      setSelectedFile(null);
      Alert.alert('Success', 'Document deleted successfully');
    } catch (error) {
      console.error('Delete error:', error.response || error);
      
      const errorMessage = error.response?.data?.details || 
                          error.response?.data?.error ||
                          error.message ||
                          'Failed to delete document';
                          
      Alert.alert('Error', errorMessage);
    }
  };

  const handleSubmit = async () => {
    if (!text.trim()) {
      Alert.alert('Error', 'Please enter some text');
      return;
    }

    setLoading(true);
    //setError('');
    setResponse('');
    
    const prompt = text;
    setText('');

    try {
      const result = await axios.get(`${localhost}/ask`, {
        params: {
          text: prompt,
          use_rag: true,
        }
      });
      
      setResponse(result.data.response);
      setRequest(prompt);
      
      await saveToHistory(prompt, result.data.response, modelName); 
    } catch (error) {
      Alert.alert('Error', error.message || 'Failed to get response');
    } finally {
      setLoading(false);
    }
  };

  const clearDatabase = () => {
    console.log('Starting database clear...');
    setShowClearConfirm(true);
  };

  const saveToHistory = async (input: string, response: string, model: string) => {
    try {
    const newItem = {
      id: Date.now().toString(),
      input,
      response,
      timestamp: new Date().toLocaleString(),
      model, // Save the model information
      isRag: true,
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
                    <Text style={styles.uploadButtonText}>Upload Document</Text>
                  </>
                )}
              </TouchableOpacity>

              <TouchableOpacity
                style={styles.clearDatabaseButton}
                onPress={clearDatabase}
                disabled={loading}
              >
                {loading ? (
                  <ActivityIndicator color="#fff" />
                ) : (
                  <FontAwesome name="trash" size={20} color="#fff" />
                )}
              </TouchableOpacity>
            </View>

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
                    <Text style={styles.fileTimestamp}>{file.timestamp}</Text>
                  </TouchableOpacity>
                ))}

                <Modal
                  isVisible={isModalVisible}
                  onBackdropPress={() => setIsModalVisible(false)}
                  style={styles.modal}
                >
                  <View style={styles.modalContent}>
                    <View style={styles.modalHeader}>
                      <Text style={styles.modalTitle}>
                        {selectedFile?.name}
                      </Text>
                      <TouchableOpacity
                        onPress={() => setIsModalVisible(false)}
                        style={styles.closeButton}
                      >
                        <FontAwesome name="times" size={20} color="#333" />
                      </TouchableOpacity>
                    </View>

                    {loadingChunks ? (
                      <ActivityIndicator size="large" color="#007AFF" />
                    ) : (
                      <ScrollView style={styles.chunksContainer}>
                        {documentChunks.map((chunk, index) => (
                          <View key={chunk.id} style={styles.chunkItem}>
                            <View style={styles.chunkHeader}>
                              <Text style={styles.chunkTitle}>Chunk {index + 1}</Text>
                              <TouchableOpacity
                                onPress={() => handleDeleteChunk(chunk.id)}
                                style={styles.deleteChunkButton}
                              >
                                <FontAwesome name="trash" size={16} color="#ff4444" />
                              </TouchableOpacity>
                            </View>
                            <Text style={styles.chunkId}>ID: {chunk.id}</Text>
                            <Text style={styles.chunkContent}>{chunk.content}</Text>
                          </View>
                        ))}
                      </ScrollView>
                    )}

                    <TouchableOpacity
                      style={styles.deleteDocumentButton}
                      onPress={handleDeleteDocument}
                    >
                      <FontAwesome name="trash" size={16} color="#fff" />
                      <Text style={styles.deleteDocumentText}>Delete Document</Text>
                    </TouchableOpacity>
                  </View>
                </Modal>
              </View>
            )}
          </View>

          <View style={styles.chatSection}>
            <TextInput
              style={[styles.input, { height: Math.max(120, inputHeight) }]}
              value={text}
              onChangeText={setText}
              placeholder="Ask about your documents..."
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
                <Text style={styles.buttonText}>Ask</Text>
              )}
            </TouchableOpacity>

            {response ? (
              <View style={styles.responseContainer}>
                <Text style={styles.responseTitle}>Your Question:</Text>
                <Text style={styles.requestText}>{request}</Text>
                <Text style={styles.responseTitle}>Answer:</Text>
                <ScrollView style={styles.responseScroll}>
                  <Text style={styles.responseText}>{response}</Text>
                </ScrollView>
              </View>
            ) : null}
          </View>
        </ScrollView>
      </KeyboardAvoidingView>

      <AlertModal
        visible={showClearConfirm}
        onClose={() => setShowClearConfirm(false)}
        title="Clear Database"
        message="Are you sure you want to clear all uploaded files?"
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
                setLoading(true);
                console.log('Making request to clear database...');
                
                const response = await axios.get(`${localhost}/clear_db`);
                
                const data = response.data;

                console.log('Clear database response:', data);
                
                // Clear local state
                setUploadedFiles([]);
                await AsyncStorage.setItem('uploadedFiles', JSON.stringify([]));

                setShowClearConfirm(false);

                if (parseInt(data.removed_count) > 0){
                  alert(`Successfully cleared ${data.removed_count} database vector chunks.`)
                } else {
                  alert(data.message);
                }
                
              } catch (error) {
                setShowClearConfirm(false);
                console.error('Failed to clear database:', error);
                alert("Failed to clear database.");
              } finally {
                setShowClearConfirm(false);
                setLoading(false);
              }
              setShowClearConfirm(false);
              setLoading(false);
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
  keyboardAvoid: {
    flex: 1,
  },
  content: {
    flex: 1,
  },
  contentContainer: {
    padding: 20,
  },
  uploadSection: {
    marginBottom: 20,
  },
  uploadButton: {
    backgroundColor: '#28a745',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 15,
    borderRadius: 8,
    marginBottom: 15,
  },
  uploadButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 10,
  },
  filesContainer: {
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 15,
    borderWidth: 1,
    borderColor: '#ddd',
  },
  filesTitle: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 10,
    color: '#333',
  },
  fileItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  fileName: {
    flex: 1,
    marginLeft: 10,
    fontSize: 14,
    color: '#333',
  },
  fileTimestamp: {
    fontSize: 12,
    color: '#666',
    marginLeft: 10,
  },
  chatSection: {
    flex: 1,
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
  modal: {
    margin: 0,
    maxWidth: "90%",
    alignSelf: "center",
    justifyContent: 'flex-end',
  },
  modalContent: {
    backgroundColor: '#fff',
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    paddingBottom: 40,
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
    fontWeight: 'bold',
    flex: 1,
  },
  closeButton: {
    padding: 5,
  },
  chunksContainer: {
    padding: 20,
  },
  chunkItem: {
    marginBottom: 20,
    padding: 15,
    backgroundColor: '#f8f9fa',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#eee',
  },
  chunkHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  chunkTitle: {
    fontSize: 16,
    fontWeight: '600',
  },
  chunkId: {
    fontSize: 12,
    color: '#666',
    marginBottom: 5,
  },
  chunkContent: {
    fontSize: 14,
    color: '#333',
    lineHeight: 20,
  },
  deleteChunkButton: {
    padding: 5,
  },
  deleteDocumentButton: {
    backgroundColor: '#ff4444',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 15,
    marginHorizontal: 20,
    borderRadius: 8,
  },
  deleteDocumentText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 10,
  },
  buttonRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 15,
  },
  uploadButton: {
    backgroundColor: '#28a745',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 15,
    borderRadius: 8,
    flex: 1,
    marginRight: 10,
  },
  clearDatabaseButton: {
    backgroundColor: '#dc3545',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 15,
    borderRadius: 8,
    width: 50,  // Make it square
    height: 50, // Make it square
  },
  uploadButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 10,
  },
});