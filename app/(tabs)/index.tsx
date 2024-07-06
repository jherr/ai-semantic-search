import { Layout, Text, Input } from "@ui-kitten/components";

export default function HomeScreen() {
  return (
    <Layout
      style={{
        flex: 1,
        justifyContent: "flex-start",
        alignItems: "flex-start",
        padding: 10,
        paddingTop: 50,
      }}
    >
      <Text category="h1">HOME</Text>
      <Input placeholder="Search" />
    </Layout>
  );
}
