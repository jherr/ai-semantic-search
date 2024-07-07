import { useEffect, useState } from "react";

import {
  Layout,
  Text,
  Input,
  Divider,
  List,
  ListItem,
} from "@ui-kitten/components";

import winkNLP from "wink-nlp";
import model from "wink-eng-lite-web-model";
import BM25Vectorizer from "wink-nlp/utilities/bm25-vectorizer";
import { HNSW } from "hnsw";

import { Restaurant, restaurants } from "@/src/restaurants";

async function buildIndex<
  T extends {
    id: number;
    title: string;
    synopsis: string;
  }
>(docs: T[]) {
  const nlp = winkNLP(model, ["pos"]);
  const its = nlp.its;

  const bm25 = BM25Vectorizer();
  for (const d of docs) {
    bm25.learn(
      nlp
        .readDoc(`${d.title} ${d.synopsis}`)
        .tokens()
        .filter((t) => t.out(its.type) !== "punctuation")
        .out(its.normal)
    );
  }

  const data: {
    id: number;
    vector: number[];
  }[] = [];
  for (const d of docs) {
    data.push({
      id: d.id,
      vector: bm25.vectorOf(
        nlp
          .readDoc(d.title)
          .tokens()
          .filter((t) => t.out(its.type) !== "punctuation")
          .out(its.normal)
      ),
    });
    data.push({
      id: d.id,
      vector: bm25.vectorOf(
        nlp
          .readDoc(d.synopsis)
          .tokens()
          .filter((t) => t.out(its.type) !== "punctuation")
          .out(its.normal)
      ),
    });
  }
  const hnsw = new HNSW(200, 16, data[0].vector.length, "cosine");
  await hnsw.buildIndex(data);

  return {
    query: (q: string, n: number = 3): T[] => {
      const vector = bm25.vectorOf(
        nlp
          .readDoc(q)
          .tokens()
          .filter((t) => t.out(its.type) !== "punctuation")
          .out(its.normal)
      );
      const found = hnsw.searchKNN(vector, n);
      return found.map((f) => docs.find((m) => m.id === f.id)) as T[];
    },
  };
}

const index = buildIndex(restaurants);

export default function HomeScreen() {
  const [search, setSearch] = useState("mexican food");
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
