// app/(tabs)/_layout.tsx
import { Tabs } from 'expo-router';
import { useColorScheme } from 'react-native';
import { FontAwesome } from '@expo/vector-icons';

// You can import your colors from a constants file if you prefer
const tintColorLight = '#2f95dc';
const tintColorDark = '#fff';

export default function TabLayout() {
  const colorScheme = useColorScheme();

  return (
    <Tabs
      screenOptions={{
        tabBarActiveTintColor: colorScheme === 'dark' ? tintColorDark : tintColorLight,
        // Tab bar style
        tabBarStyle: {
          height: 60,
          paddingBottom: 5,
        },
        // Header style
        headerStyle: {
          backgroundColor: colorScheme === 'dark' ? '#000' : '#fff',
        },
        headerTintColor: colorScheme === 'dark' ? '#fff' : '#000',
      }}>
      
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

      <Tabs.Screen
        name="chat"
        options={{
          title: 'Chat',
          tabBarIcon: ({ color }) => (
            <FontAwesome name="comments" size={28} color={color} />
          ),
          headerTitle: 'Chat Session',
        }}
      />

      <Tabs.Screen
        name="two"
        options={{
          title: 'History',
          tabBarIcon: ({ color }) => (
            <FontAwesome name="history" size={28} color={color} />
          ),
          headerTitle: 'Conversations',
        }}
      />

      <Tabs.Screen
        name="rag"
        options={{
          title: 'Docs',
          tabBarIcon: ({ color }) => (
            <FontAwesome name="file-text" size={28} color={color} />
          ),
          headerTitle: 'Document Chat',
        }}
      />
    </Tabs>
  );
}

// Helper function for tab bar icons
function TabBarIcon(props: {
  name: React.ComponentProps<typeof FontAwesome>['name'];
  color: string;
}) {
  return <FontAwesome size={28} style={{ marginBottom: -3 }} {...props} />;
}
