// app/context/AppContext.tsx
import React, { createContext, useContext, useState, ReactNode } from 'react';

interface AppContextType {
  modelName: string;
  setModelName: (name: string) => void;
  isModelLoading: boolean;  // Add this
  logThis: (label: string, data: any) => void;
}

const defaultContext: AppContextType = {
  modelName: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
  setModelName: () => {},
  logThis: () => {},
};

const AppContext = createContext<AppContextType>(defaultContext);

export const useAppContext = () => useContext(AppContext);

interface AppProviderProps {
  children: ReactNode;
}

export const AppProvider: React.FC<AppProviderProps> = ({ children }) => {
  const [modelName, setModelName] = useState(defaultContext.modelName);

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