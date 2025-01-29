// AppContext.tsx
// This file implements the Context API pattern for global state management
// It provides a way to share state between components without prop drilling

import React, { createContext, useContext, useState, ReactNode } from 'react';

interface AppContextType {
  modelName: string;        // Stores the currently selected LLM model
  setModelName: (name: string) => void;  // Function to update the model
  logThis: (label: string, data: any) => void;  // Utility for debugging
}

// Default values for the context
const defaultContext: AppContextType = {
  modelName: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',  // Default model
  setModelName: () => {},  // Empty function placeholder
  logThis: () => {},      // Empty logging placeholder
};

// Create the context with default values
const AppContext = createContext<AppContextType>(defaultContext);

// Custom hook for easy context consumption in components
export const useAppContext = () => useContext(AppContext);

interface AppProviderProps {
  children: ReactNode;  // Types the children prop as React nodes
}

// Provider component that wraps the app and provides context values
export const AppProvider: React.FC<AppProviderProps> = ({ children }) => {
  // State for the model name with the default value
  const [modelName, setModelName] = useState(defaultContext.modelName);

  // Utility function for consistent logging across the app
  const logThis = (label: string, data: any) => {
    console.log(`[${label}]`, data);
  };

  // Provide the context values to all child components
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

// AlertModal.tsx
// Reusable modal component for displaying alerts and confirmations
import React from 'react';
import { Modal, View, Text, TouchableOpacity, StyleSheet } from 'react-native';

interface Button {
  text: string;           // Button label
  onPress: () => void;    // Button action
  style?: 'default' | 'cancel' | 'destructive';  // Visual style
}

interface AlertModalProps {
  visible: boolean;       // Controls modal visibility
  onClose: () => void;   // Handler for modal dismissal
  title: string;         // Modal title
  message: string;       // Modal content
  buttons: Button[];     // Array of action buttons
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
                style={[
                  styles.button,
                  // Apply different styles based on button type
                  button.style === 'destructive' && styles.destructiveButton,
                  button.style === 'cancel' && styles.cancelButton,
                ]}
                onPress={button.onPress}
              >
                <Text
                  style={[
                    styles.buttonText,
                    button.style === 'destructive' && styles.destructiveText,
                    button.style === 'cancel' && styles.cancelText,
                  ]}
                >
                  {button.text}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>
      </View>
    </Modal>
  );
};

// ThemedView.tsx
// Component that automatically applies the current theme colors
import { View, ViewProps } from 'react-native';
import { useThemeColor } from '@/hooks/useThemeColor';

export type ThemedViewProps = ViewProps & {
  lightColor?: string;   // Optional custom light theme color
  darkColor?: string;    // Optional custom dark theme color
};

export function ThemedView({ 
  style, 
  lightColor, 
  darkColor, 
  ...otherProps 
}: ThemedViewProps) {
  // Get the appropriate background color based on current theme
  const backgroundColor = useThemeColor(
    { light: lightColor, dark: darkColor }, 
    'background'
  );

  return <View style={[{ backgroundColor }, style]} {...otherProps} />;
}
