import { useEffect, useState } from "react";

import {
  Layout,
  Text,
  Input,
  Divider,
  List,
  ListItem,
} from "@ui-kitten/components";
import { Asset } from "expo-asset";

import { HNSW } from "hnsw";
import { InferenceSession, Tensor } from "onnxruntime-react-native";
import { PreTrainedTokenizer } from "@xenova/transformers/src/tokenizers";

import { Restaurant, restaurants } from "@/src/restaurants";

// THe hidden state of the model is a 3D tensor with dimensions [batch_size, sequence_length, vector_size]
// Because we only send in one sequence at a time, the batch size is 1
// We want to average the hidden states across the sequence length to get a single vector
function globalAverage(data: number[], dims: number[]) {
  const vectorSize = dims[2];
  const average = new Array(vectorSize).fill(0);
  for (const index in data) {
    average[+index % vectorSize] += data[index];
  }
  const sequenceSize = dims[1];
  for (const index in average) {
    average[index] /= sequenceSize;
  }
  return average;
}

async function buildIndex<
  T extends {
    id: number;
    title: string;
    synopsis: string;
  }
>(docs: T[]) {
  // Load up the ONNX embedding model
  const modelPath = require("../../assets/Snowflake.onnx");
  const assets = await Asset.loadAsync(modelPath);
  const modelUri = assets[0].localUri;
  const session = await InferenceSession.create(modelUri!);

  // Configure the tokenizer with the configuration that is synced with the model
  const config = require("../../assets/tokenizer.json");
  const options = require("../../assets/tokenizer_config.json");
  const tokenizer = new PreTrainedTokenizer(config, options);

  async function getEmbedding(input: string) {
    // Convert the text into tensors that the model can understand
    const { input_ids, attention_mask } = tokenizer(input);

    // Run the model with the tensors
    const { last_hidden_state } = await session.run({
      input_ids: input_ids,
      attention_mask: attention_mask,
      token_type_ids: new Tensor(
        "int64",
        new BigInt64Array(input_ids.dims[1]),
        input_ids.dims
      ),
    });

    // Take the average of the last hidden state to get the embedding
    return globalAverage(last_hidden_state.cpuData, last_hidden_state.dims);
  }

  // Create an array of vectors for the HNSW index
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

  // Build the index
  const hnsw = new HNSW(200, 16, 384, "cosine");
  await hnsw.buildIndex(data);

  return {
    query: async (q: string, n: number = 3): Promise<T[]> => {
      // Vectorize the query
      const vector = await getEmbedding(q);

      // Search the index for that vector
      const found = hnsw.searchKNN(vector, 20);

      // Sort the results by score and return the top n
      const sorted = found.sort((a, b) => b.score - a.score).slice(0, n);

      // Return the documents that match the ids
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
