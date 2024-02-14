package luh.jnn;

import luh.jnn.nn.*;

public class Main {
  
  public static void main(String[] args) {
    Layer[] layers = new Layer[] {
      new Layer(10, 0),
          new Layer(6, 0),
          new Layer(8, 0),
          new Layer(3, 0)
    };
    NeuralNetwork nn = new NeuralNetwork(layers);
  }
}
