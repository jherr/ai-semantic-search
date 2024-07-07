import {
  Layout,
  Text,
  Input,
  Divider,
  List,
  ListItem,
} from "@ui-kitten/components";

import { useEffect, useState } from "react";

import { HNSW } from "hnsw";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-react-native";

type Restaurant = {
  id: number;
  title: string;
  synopsis: string;
};

const restaurants: Restaurant[] = [
  {
    id: 1,
    title: "Casa Zoraya",
    synopsis:
      "Since Zoraya Zambrano and her children, Gary and Gloria Marmanillo, opened Casa Zoraya back in 2018, this Peruvian spot has been Piedmont’s under-the-radar gem: Ceviches land at the table like a work of art, fried calamari adding crunch to a bed of fresh seasonal seafood tossed with a summery leche de tigre. Arroz Chaufa, a Peruvian fried rice dish, gets an upgrade with a passionfruit rocoto sauce, which adds dose of acidity, sweetness, and peppery zip. And the pisco sours feel like they’re shipped straight from Lima, best sipped on Casa Zoraya’s back patio on nice days.",
  },
  {
    id: 2,
    title: "Hat Yai",
    synopsis:
      "Akkapong Earl Ninsom did it again: After treating Portland to deftly executed Thai cooking at Langbaan and Paadee, the chef —alongside co-founder and co-owner Alan Akwai — created a casual southern Thai compatriot on Northeast Killingsworth with hardcore devotees. Hat Yai’s shallot-fried chicken, salty and crunchy, pairs exceptionally well with Malayu-style curry and crispy roti, all available in the popular combo No. 1. And yet, diners will be rewarded for straying from the top billing: The restaurant’s searingly spicy kua gling ground pork is abundant with aromatics and alliums, and the dtom som shrimp combines seafood with meaty oyster mushrooms in a broth pleasingly sour with tamarind and ginger.",
  },
  {
    id: 3,
    title: "Gabbiano's",
    synopsis:
      "In a period of time where so many Portland restaurants are overwhelmingly earnest, Killingsworth neighborhood Italian American joint Gabbiano’s exudes a good-hearted ridiculousness, a commitment to the bit that feels truly refreshing. Fried mozzarella shot glasses filled with marinara? For sure. Caprese Negronis with sundried tomato Campari and mozzarella ball garnishes? Totally. But if Gabbiano’s were simply a gimmick, it wouldn’t appear on this map; each dish has a true sense of deliciousness, from frisbee-sized discs of juicy chicken Parm in a bright pomodoro, or a Dungeness crab alla vodka with the heat and crunch of pistachio chile crisp. Not every restaurant needs to be serious; this city needs its silly little corners, and Gabbiano’s is one of them.",
  },
  {
    id: 4,
    title: "Baon Kainan",
    synopsis:
      "Ethan and Geri Leung went from popping up in Seattle to opening this casual Alberta food cart, which offers a simultaneously inventive and accessible take on Filipino staples. Every dish has an incredible depth of flavor, whether it’s the lingering floral brightness of calamansi in a rich roasted pork sisig, or the tamari-laden adobo, which hits the grill for a touch of char and smoke. Brunches include sticky glazed tocino and satisfyingly simple garlic rice, each dish popping with acid and salt.",
  },
  {
    id: 5,
    title: "Pasture PDX",
    synopsis:
      "Farm-to-table is likely the most pervasive of the Portland culinary cliches; every restaurant in town touts some version of its ethos, name-dropping a few farms or grabbing a few seasonal items from the farmers market. But Pasture owners Kei Ohdera and HJ Schaible take their emphasis on responsible sourcing to a new level, seeking out and developing relationships with regenerative farms for whole-animal butchery inside the restaurant. The result: Straight-up delicious sandwiches, ranging from beef mortadella to achingly tender pastrami, served with house pickled peppers. It’s not just walking the walk of sustainability; it’s making it feel approachable (and delicious).",
  },
  {
    id: 6,
    title: "Mole Mole Mexican Cuisine",
    synopsis:
      "Each day, during lunch hours, Alberta locals line up at this orange-and-green cart in pursuit of chiles en nogada stuffed with ground pork and bowls of lipstick-red pozole, sipping prickly pear agua fresca and horchata while they wait for their orders. The cart’s menu is extensive, with everything from soy curl burritos to cochinita pibil, but it should be no surprise that this cart’s particular specialty is its moles: a sweet and nutty mole negro, an herbaceous and vegetal mole verde. The cart’s fuchsia mole rosa, a rarity at Portland Mexican restaurants made with earthy beets and hibiscus flowers, is available as a coating for tender enchiladas or simply paired with fresh salmon, a smart choice of protein for the sweet-earthy beet sauce. The artful plating — colorful ceramic bowls, garnished with flowers — sets each dish over the top, making this one of the city’s finest Mexican carts.",
  },
  {
    id: 7,
    title: "Urdaneta",
    synopsis:
      "At this intimate pintxo bar, Javier and Jael Canteras have developed a reputation for straight-up goofy dishes winking at Northern Spanish flavors, like a bikini (ham and cheese sandwich) made with American and jamon serrano, or an octopus a la brasa with chorizo XO. That being said, the traditional Spanish dishes on the menu remain true to the originals, whether it’s crispy-on-the-outside, gooey-on-the-inside croquetas de jamon, or a blackened slice of Basque cheesecake. The restaurant’s selection of vermouth and sherry would make any Iberian proud.",
  },
  {
    id: 8,
    title: "Lovely's Fifty Fifty",
    synopsis:
      "In a North Mississippi pizza cafe that feels casual but intimate, pizzaiola Sarah Minnick embraces paradoxes: She took something brimming with childhood charm — pizza and ice cream — and gave it a high-end twist. Ever-changing pizzas are a garden of edible flowers and mushrooms, atop an airy-but-sturdy pizza dough made with Oregon whole grains; they’re joined by salads and soups made with peak-season produce. While the menu changes on an almost weekly basis, Minnick’s culinary creativity and attention to detail remains constant. If someone is defining Portland’s distinct pizza style, it’s Minnick.",
  },
];

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
      console.log(sorted);
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
