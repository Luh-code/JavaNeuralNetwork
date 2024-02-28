package luh.jnn.nn;

import java.io.Serializable;

import luh.jnn.Logging;

public class Layer implements Serializable {
  private static final long serialVersionUID = 1L;
  private Neuron[] neurons;
  private float bias;

  public Layer(Neuron[] neurons, float bias) {
    this.neurons = neurons;
    this.bias = bias;
  }

  public Layer(int count, float bias) {
    if (count < 0) {
      Logging.logger.fatal("Neuron count cannot be smaller than 0");
      System.exit(1);
    }
    this.bias = bias;

    this.neurons = new Neuron[count];

    for (int i = 0; i < this.neurons.length; ++i) {
      this.neurons[i] = new Neuron();
    }
  }

  public Neuron[] getNeurons() {
    return this.neurons;
  }

  public void setNeurons(Neuron[] neurons) {
    this.neurons = neurons;
  }

  public Neuron getNeuron(int index) {
    return getNeurons()[index];
  }

  public int getTensorSize() {
    return this.neurons.length;
  }

  public void setTensor(float[] tensor) {
    if (tensor.length < getTensorSize()) {
      for (Neuron n : neurons) {
        n.setZ(0);
      }
    }
    for (int i = 0; i < Math.min(tensor.length, getTensorSize()); ++i) {
      neurons[i].setZ(tensor[i]);
    }
  }

  public float[] getTensor() {
    float[] tensor = new float[getTensorSize()];
    for(int i = 0; i < getTensorSize(); ++i) {
      tensor[i] = getNeuron(i).getZ();
    }
    return tensor;
  }

  public float getBias() {
    return bias;
  }

  public void setBias(float bias) {
    this.bias = bias;
  }
}
