// components/modal.tsx
import React from 'react';
import {
  Modal,
  View,
  Text,
  TouchableOpacity,
  Platform,
  StyleSheet,
  Pressable,
} from 'react-native';

const AlertModal = ({
  visible,
  onClose,
  title,
  message,
  buttons = [{ text: 'OK', style: 'default' }]
}) => {
  return (
    <Modal
      visible={visible}
      transparent={true}
      animationType="fade"
      onRequestClose={onClose}
    >
      <Pressable 
        style={styles.backdrop} 
        onPress={onClose}
      >
        <View 
          style={styles.modalContainer}
          // Stop propagation of touch events to prevent closing when clicking modal
          onStartShouldSetResponder={() => true}
          onTouchEnd={e => e.stopPropagation()}
        >
          {/* Header */}
          <View style={styles.header}>
            <Text style={styles.title}>{title}</Text>
          </View>

          {/* Content */}
          <View style={styles.content}>
            <Text style={styles.message}>{message}</Text>
          </View>

          {/* Buttons */}
          <View style={styles.buttonContainer}>
            {buttons.map((button, index) => (
              <TouchableOpacity
                key={index}
                style={[
                  styles.button,
                  button.style === 'destructive' && styles.destructiveButton,
                  button.style === 'cancel' && styles.cancelButton,
                  index > 0 && styles.buttonMargin
                ]}
                onPress={() => {
                  button.onPress?.();
                  if (button.style !== 'cancel') {
                    onClose();
                  }
                }}
              >
                <Text
                  style={[
                    styles.buttonText,
                    button.style === 'destructive' && styles.destructiveButtonText,
                    button.style === 'cancel' && styles.cancelButtonText,
                  ]}
                >
                  {button.text}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>
      </Pressable>
    </Modal>
  );
};

const styles = StyleSheet.create({
  backdrop: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
    ...Platform.select({
      web: {
        cursor: 'default',
      },
    }),
  },
  modalContainer: {
    backgroundColor: '#fff',
    borderRadius: 12,
    width: '90%',
    maxWidth: 500,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.25,
    shadowRadius: 4,
    elevation: 5,
    ...Platform.select({
      web: {
        maxHeight: '90vh',
      },
    }),
  },
  header: {
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
    padding: 16,
  },
  title: {
    fontSize: 18,
    fontWeight: '600',
    color: '#000',
  },
  content: {
    padding: 16,
  },
  message: {
    fontSize: 16,
    color: '#333',
    lineHeight: 24,
  },
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    padding: 16,
    backgroundColor: '#f8f8f8',
    borderBottomLeftRadius: 12,
    borderBottomRightRadius: 12,
  },
  button: {
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 8,
    backgroundColor: '#007AFF',
    minWidth: 80,
    alignItems: 'center',
  },
  buttonMargin: {
    marginLeft: 8,
  },
  destructiveButton: {
    backgroundColor: '#FF3B30',
  },
  cancelButton: {
    backgroundColor: '#E5E5EA',
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '500',
  },
  destructiveButtonText: {
    color: '#fff',
  },
  cancelButtonText: {
    color: '#000',
  },
});

export default AlertModal;
