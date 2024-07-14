import { useEffect, useState } from "react";

import {
  Layout,
  Text,
  Input,
  Divider,
  List,
  ListItem,
} from "@ui-kitten/components";

import { Restaurant } from "@/src/restaurants";

import { searchRestaurants } from "@/src/semantic-search";

export default function HomeScreen() {
  const [search, setSearch] = useState("asian fusion");
  const [results, setResults] = useState<Restaurant[]>([]);
  useEffect(() => {
    (async () => {
      setResults(await searchRestaurants(search));
    })();
  }, [search]);

  const renderItem = ({
    item,
  }: {
    item: Restaurant;
    index: number;
  }): React.ReactElement => (
    <ListItem title={item.name} description={item.description} />
  );

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
      <Text category="h1">Search</Text>
      <Input placeholder="Search" value={search} onChangeText={setSearch} />
      <List
        style={{ maxHeight: 1200, width: "100%" }}
        data={results}
        ItemSeparatorComponent={Divider}
        renderItem={renderItem}
      />
    </Layout>
  );
}
