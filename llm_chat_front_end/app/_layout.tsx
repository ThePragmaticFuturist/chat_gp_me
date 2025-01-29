// app/_layout.tsx (Root layout)
import { AppProvider } from './context/AppContext';
import { Stack } from 'expo-router';
import { useColorScheme } from 'react-native';
import ErrorBoundary from './ErrorBoundary';

export default function RootLayout() {
  const colorScheme = useColorScheme();
//<Stack.Screen name="modal" options={{ presentation: 'modal' }} />
  return (
    <AppProvider>
      <ErrorBoundary>
        <Stack>
          <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
        </Stack>
      </ErrorBoundary>
    </AppProvider>
  );
}