// app/index.tsx (Root index)
import { Redirect } from 'expo-router';

export default function Index() {
  // Redirect to the first tab
  return <Redirect href="/(tabs)" />;
}
