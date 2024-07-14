import { Asset } from "expo-asset";
import { HNSW } from "hnsw";
import { InferenceSession, Tensor } from "onnxruntime-react-native";
import { PreTrainedTokenizer } from "@xenova/transformers/src/tokenizers";

import { restaurants } from "@/src/restaurants";
import { globalAverage, zeroTensor } from "./utils";

import tConfig from "./tokenizer.json";
import tOptions from "./tokenizer_config.json";

// Initialize the tokenizer
const tokenizer = new PreTrainedTokenizer(tConfig, tOptions);

// Create an embedding function and an HNSW index of restaurants
async function buildIndex() {
  // Load up the ONNX embedding model
  const modelPath = require("./Snowflake.onnx");
  const assets = await Asset.loadAsync(modelPath);
  const modelUri = assets[0].localUri;
  const session = await InferenceSession.create(modelUri!);

  async function createEmbedding(input: string) {
    // Convert the text into tensors that the model can understand
    const { input_ids, attention_mask } = tokenizer(input);

    // Run the model with the tensors
    const { last_hidden_state } = await session.run({
      input_ids,
      attention_mask,
      token_type_ids: zeroTensor(input_ids.dims),
    });

    // Take the average of the last hidden state to get the embedding
    return globalAverage(last_hidden_state.cpuData, last_hidden_state.dims);
  }

  // Create an array of vectors for the HNSW index
  const rows = [];
  for (const r of restaurants) {
    rows.push({
      id: r.id,
      vector: await createEmbedding(r.description),
    });
  }

  // Build the index
  const hnsw = new HNSW(200, 16, rows[0].vector.length, "cosine");
  await hnsw.buildIndex(rows);

  return { hnsw, createEmbedding };
}

const index = buildIndex();

export async function searchRestaurants(q: string, n: number = 5) {
  // Get the inference session and the index
  const { createEmbedding, hnsw } = await index;

  // Vectorize the query
  const vector = await createEmbedding(q);

  // Search the index for that vector
  const found = hnsw.searchKNN(vector, n * 4);

  // Sort the results by score and return the top n
  const sorted = found.sort((a, b) => b.score - a.score).slice(0, n);

  // Return the documents that match the ids
  return sorted.map((f) => restaurants.find((m) => m.id === f.id));
}
