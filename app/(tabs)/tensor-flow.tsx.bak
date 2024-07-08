import { useEffect, useState } from "react";

import {
  Layout,
  Text,
  Input,
  Divider,
  List,
  ListItem,
} from "@ui-kitten/components";

import { HNSW } from "hnsw";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-react-native";

import { Restaurant, restaurants } from "@/src/restaurants";

async function buildIndex<
  T extends {
    id: number;
    title: string;
    synopsis: string;
  }
>(docs: T[]) {
  const tensorFlowStartup = Date.now();
  await tf.ready();
  console.log(`tensorFlowStartup: ${Date.now() - tensorFlowStartup}ms`);

  const loadEncoder = Date.now();
  const use = require("@tensorflow-models/universal-sentence-encoder");
  const model = await use.load();
  console.log(`loadEncoder: ${Date.now() - loadEncoder}ms`);

  const createDocs = Date.now();
  const docsById = docs.reduce((acc, doc) => {
    acc[doc.id] = doc;
    return acc;
  }, {} as Record<number, T>);
  console.log(`createDocs: ${Date.now() - createDocs}ms`);

  const createEmbeddings = Date.now();
  const embeddingsTensor = await model.embed(
    docs.map((d) => `${d.title} ${d.synopsis}`)
  );
  const embeddings = embeddingsTensor.arraySync();
  console.log(`createEmbeddings: ${Date.now() - createEmbeddings}ms`);

  const createTreeData = Date.now();
  const data = [];
  for (const docIndex in docs) {
    const doc = docs[docIndex];
    const vector = embeddings[docIndex];
    data.push({ id: doc.id, vector });
  }
  console.log(`createTreeData: ${Date.now() - createTreeData}ms`);

  const createHSNW = Date.now();
  const hnsw = new HNSW(200, 16, 512, "cosine");
  await hnsw.buildIndex(data);
  console.log(`createHSNW: ${Date.now() - createHSNW}ms`);

  return {
    query: async (q: string, n: number = 3): Promise<T[]> => {
      if (q.trim() === "") return [];
      const queryEmbed = Date.now();
      const queryTensor = await model.embed([q]);
      const vector = queryTensor.arraySync()[0];
      console.log(`queryEmbed: ${Date.now() - queryEmbed}ms`);

      const searchKNN = Date.now();
      const found = hnsw.searchKNN(vector, n);
      console.log(`searchKNN: ${Date.now() - searchKNN}ms`);

      const sorted = found.sort((a, b) => (a.score > b.score ? -1 : 1));
      return sorted.map((f) => docsById[f.id]) as T[];
    },
  };
}

let index: ReturnType<typeof buildIndex<Restaurant>> | null = null;

export default function HomeScreen() {
  const [search, setSearch] = useState("latin");
  const [results, setResults] = useState<Restaurant[]>([]);
  useEffect(() => {
    (async () => {
      if (!index) index = buildIndex(restaurants);
      const { query } = await index;
      setResults(await query(search));
    })();
  }, [search]);

  const renderItem = ({
    item,
  }: {
    item: Restaurant;
    index: number;
  }): React.ReactElement => (
    <ListItem title={item.title} description={item.synopsis} />
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
        style={{ maxHeight: 500, width: "100%" }}
        data={results}
        ItemSeparatorComponent={Divider}
        renderItem={renderItem}
      />
    </Layout>
  );
}
