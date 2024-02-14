package luh.jnn.nn;

public class Layer {
  private Neuron[] neurons;

  public Layer(Neuron[] neurons) {
    this.neurons = neurons;
  }
  public Layer(int count, float startingBias) {
    if (count < 0) {
      throw new RuntimeException("Neuron count cannot be smaller than 0");
    }

    this.neurons = new Neuron[count];

    for (int i = 0; i < this.neurons.length; ++i) {
      this.neurons[i] = new Neuron(startingBias);
    }
  }

  public Neuron[] getNeurons() {
    return this.neurons;
  }

  public void setNeurons(Neuron[] neurons) {
    this.neurons = neurons;
  }
}
