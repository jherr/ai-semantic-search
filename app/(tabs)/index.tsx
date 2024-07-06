import {
  Layout,
  Text,
  Input,
  Divider,
  List,
  ListItem,
} from "@ui-kitten/components";
import { StyleSheet } from "react-native";

import { useEffect, useState } from "react";

import winkNLP from "wink-nlp";
import model from "wink-eng-lite-web-model";
import BM25Vectorizer from "wink-nlp/utilities/bm25-vectorizer";
import { HNSW } from "hnsw";

type Movie = {
  id: number;
  title: string;
  synopsis: string;
};

const movies: Movie[] = [
  {
    id: 1,
    title: "Aliens",
    synopsis:
      "Ripley, the sole survivor of the Nostromo's deadly encounter with the monstrous Alien, returns to Earth after drifting through space in hypersleep for 57 years. Although her story is initially met with skepticism, she agrees to accompany a team of Colonial Marines back to LV-426.",
  },
  {
    id: 2,
    title: "The Terminator",
    synopsis:
      "In the post-apocalyptic future, reigning tyrannical supercomputers teleport a cyborg assassin known as the 'Terminator' back to 1984 to kill Sarah Connor, whose unborn son is destined to lead insurgents against 21st-century mechanical hegemony. Meanwhile, the human-resistance movement dispatches a lone warrior to safeguard Sarah.",
  },
  {
    id: 3,
    title: "Alien",
    synopsis:
      "During its return to the earth, commercial spaceship Nostromo intercepts a distress signal from a distant planet. When a three-member team of the crew discovers a chamber containing thousands of eggs on the planet, a creature inside one of the eggs attacks an explorer. The entire crew is unaware of the impending nightmare set to descend upon them when the alien parasite planted inside its unfortunate host is birthed.",
  },
  {
    id: 4,
    title: "The Matrix",
    synopsis:
      "Set in the 22nd century, The Matrix tells the story of a computer hacker who joins a group of underground insurgents fighting the vast and powerful computers who now rule the earth.",
  },
  {
    id: 5,
    title: "The Thing",
    synopsis:
      "Scientists in the Antarctic are confronted by a shape-shifting alien that assumes the appearance of the people that it kills.",
  },
  {
    id: 6,
    title: "Blade Runner",
    synopsis:
      "In the smog-choked dystopian Los Angeles of 2019, blade runner Rick Deckard is called out of retirement to terminate a quartet of replicants who have escaped to Earth seeking their creator for a way to extend their short life spans.",
  },
  {
    id: 7,
    title: "The Fly",
    synopsis: "A brilliant but eccentric scientist begins to transform",
  },
];

async function buildIndex<
  T extends {
    id: number;
    title: string;
    synopsis: string;
  }
>(docs: T[]) {
  const nlp = winkNLP(model);
  const its = nlp.its;

  const bm25 = BM25Vectorizer();
  for (const d of docs) {
    bm25.learn(
      nlp.readDoc(`${d.title} ${docs.synopsis}`).tokens().out(its.normal)
    );
  }

  const data: {
    id: number;
    vector: number[];
  }[] = [];
  for (const d of docs) {
    data.push({
      id: d.id,
      vector: bm25.vectorOf(nlp.readDoc(d.title).tokens().out(its.normal)),
    });
    data.push({
      id: d.id,
      vector: bm25.vectorOf(nlp.readDoc(d.synopsis).tokens().out(its.normal)),
    });
  }
  const hnsw = new HNSW(200, 16, data[0].vector.length, "cosine");
  await hnsw.buildIndex(data);

  return {
    query: (q: string, n: number = 3): T[] => {
      const vector = bm25.vectorOf(nlp.readDoc(q).tokens().out(its.normal));
      const found = hnsw.searchKNN(vector, n);
      return found.map((f) => movies.find((m) => m.id === f.id)) as T[];
    },
  };
}

const index = buildIndex(movies);

export default function HomeScreen() {
  const [search, setSearch] = useState("Alien");
  const [results, setResults] = useState<(typeof movies)[number][]>([]);
  useEffect(() => {
    (async () => {
      const { query } = await index;
      setResults(query(search));
    })();
  }, [search]);

  const renderItem = ({
    item,
  }: {
    item: Movie;
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
