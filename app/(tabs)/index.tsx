import { Layout, Text, Input } from "@ui-kitten/components";

import { useEffect } from "react";

import winkNLP from "wink-nlp";
import model from "wink-eng-lite-web-model";
import BM25Vectorizer from "wink-nlp/utilities/bm25-vectorizer";
import { HNSW } from "hnsw";

const hnsw = new HNSW(200, 16, 5, "cosine");
const data = [
  { id: 1, vector: [1, 2, 3, 4, 5] },
  { id: 2, vector: [2, 3, 4, 5, 6] },
  { id: 3, vector: [3, 4, 5, 6, 7] },
  { id: 4, vector: [4, 5, 6, 7, 8] },
  { id: 5, vector: [5, 6, 7, 8, 9] },
];

hnsw.buildIndex(data).then(() => {
  const results = hnsw.searchKNN([6, 7, 8, 9, 10], 2);
  console.log(results);
});

const nlp = winkNLP(model);
const its = nlp.its;

export default function HomeScreen() {
  useEffect(() => {
    (async () => {
      const bm25 = BM25Vectorizer();
      const corpus = [
        "Bach",
        "J Bach",
        "Johann S Bach",
        "Johann Sebastian Bach",
        "Mozart",
        "Amadeus Mozart",
        "Beethoven",
      ];
      corpus.forEach((doc) =>
        bm25.learn(nlp.readDoc(doc).tokens().out(its.normal))
      );
      // bm25.consolidate();

      console.log(
        bm25.vectorOf(
          nlp.readDoc("Johann Bach symphony").tokens().out(its.normal)
        )
      );

      console.log(
        bm25.vectorOf(nlp.readDoc("Amadeus").tokens().out(its.normal))
      );
    })();
  }, []);

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
