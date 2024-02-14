package luh.jnn.nn;

public class NeuralNetwork {
  private Layer[] layers;

  public NeuralNetwork(Layer[] layers) {
    this.layers = layers;
  }

  public Layer[] getLayers() {
    return this.layers;
  }

  public Layer getLayer(int index) {
    if (index >= this.layers.length) {
      throw new RuntimeException(String.format("Cannot get Layer with index '%d', as '%d' is the maximum index", index, this.layers.length-1));
    }
    return this.layers[index];
  }
}
