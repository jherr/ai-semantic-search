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
import { InferenceSession, Tensor } from "onnxruntime-react-native";
import { Asset } from "expo-asset";
import { PreTrainedTokenizer } from "@xenova/transformers/src/tokenizers";

import { Restaurant, restaurants } from "@/src/restaurants";

async function buildIndex<
  T extends {
    id: number;
    title: string;
    synopsis: string;
  }
>(docs: T[]) {
  const modelPath = require("./Snowflake.onnx");
  const assets = await Asset.loadAsync(modelPath);
  const modelUri = assets[0].localUri;
  const session = await InferenceSession.create(modelUri!);

  const config = require("./tokenizer.json");
  const options = require("./tokenizer_config.json");
  const tokenizer = new PreTrainedTokenizer(config, options);

  async function getEmbedding(input: string) {
    try {
      const { input_ids, attention_mask } = tokenizer(input);

      const { last_hidden_state } = await session.run({
        input_ids: input_ids,
        attention_mask: attention_mask,
        token_type_ids: new Tensor(
          "int64",
          new BigInt64Array(input_ids.dims[1]),
          input_ids.dims
        ),
      });

      const vectorSize = last_hidden_state.dims[2];
      const average = new Array(vectorSize).fill(0);
      for (const index in last_hidden_state.cpuData) {
        average[+index % vectorSize] += last_hidden_state.cpuData[index];
      }
      const sequenceSize = last_hidden_state.dims[1];
      for (const index in average) {
        average[index] /= sequenceSize;
      }
      return average;
    } catch (e) {
      console.log(e);
      return [];
    }
  }

  const data = [];
  for (const restaurant of restaurants) {
    data.push({
      id: restaurant.id,
      vector: await getEmbedding(restaurant.title),
    });
    data.push({
      id: restaurant.id,
      vector: await getEmbedding(restaurant.synopsis),
    });
  }

  const hnsw = new HNSW(200, 16, 384, "cosine");
  await hnsw.buildIndex(data);

  return {
    query: async (q: string, n: number = 3): Promise<T[]> => {
      const vector = await getEmbedding(q);
      const found = hnsw.searchKNN(vector, 20);
      const sorted = found.sort((a, b) => b.score - a.score).slice(0, n);
      return sorted.map((f) => docs.find((m) => m.id === f.id)) as T[];
    },
  };
}

const index = buildIndex(restaurants);

export default function HomeScreen() {
  const [search, setSearch] = useState("asian fusion");
  const [results, setResults] = useState<Restaurant[]>([]);
  useEffect(() => {
    (async () => {
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
