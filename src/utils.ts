import { Tensor } from "onnxruntime-react-native";

export function globalAverage(data: number[], dims: number[]) {
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

export const zeroTensor = (dims: number[]) =>
  new Tensor("int64", new BigInt64Array(dims[1]), dims);
