package luh.jnn.nn;

import java.io.Serializable;
import java.util.Arrays;

import luh.jnn.Logging;

public class Layer implements Serializable, Cloneable {
  private static final long serialVersionUID = 1L;
  private Neuron[] neurons;
  private float[] bias;
  private ActivationFunction activationFunction;

  public Layer(Neuron[] neurons, float[] bias, ActivationFunction activationFunction) {
    this.neurons = neurons;
    this.bias = bias;
    this.activationFunction = activationFunction;
  }

  public Layer(Neuron[] neurons, float initialBias, ActivationFunction activationFunction) {
    this.neurons = neurons;
    this.bias = new float[this.neurons.length];
		Arrays.fill(this.bias, initialBias);
    this.activationFunction = activationFunction;
  }

  public Layer(int count, float initialBias, ActivationFunction activationFunction) {
    if (count < 0) {
      Logging.logger.fatal("Neuron count cannot be smaller than 0");
      System.exit(1);
    }
    this.neurons = new Neuron[count];
    for (int i = 0; i < this.neurons.length; ++i) {
      this.neurons[i] = new Neuron();
    }

    this.bias = new float[this.neurons.length];
    Arrays.fill(this.bias, initialBias);

    this.activationFunction = activationFunction;
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
    if (tensor.length != getTensorSize()) {
      throw new RuntimeException("sdfsf");
    }
//    if (tensor.length < getTensorSize()) {
//      for (Neuron n : neurons) {
//        n.setZ(0);
//      }
//    }
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
    return this.bias[0];
  }

  public void setBias(float bias) {
    Arrays.fill(this.bias, bias);
  }

  public float[] getBiases() {
    return this.bias;
  }

  public void setBiases(float[] bias) {
    this.bias = bias;
  }

  public ActivationFunction getActivationFunction() {
    return activationFunction;
  }

  public void setActivationFunction(ActivationFunction activationFunction) {
    this.activationFunction = activationFunction;
  }

  @Override
  public Layer clone() {
    Neuron[] tempNeurons = new Neuron[this.neurons.length];
    for(int i = 0; i < tempNeurons.length; i++) {
      tempNeurons[i] = this.neurons[i].clone();
    }
    Layer temp = new Layer(tempNeurons, this.bias, this.activationFunction);
    return temp;
  }
}
