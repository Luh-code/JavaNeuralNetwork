package luh.jnn.nn;

import java.io.Serializable;

public class NeuralNetwork implements Serializable {
  private static final long serialVersionUID = 1L;
  private Layer[] layers;

  public NeuralNetwork(Layer[] layers) {
    this.layers = layers;
  }

  public void initalizeDenseNeuralNetwork() {
    for (int i = 1; i < getLayerCount(); ++i) {
      createDenseSynapticConnection(i-1, i, 1.0f);
    }
  }

  private void createDenseSynapticConnection(int layerIIndex, int layerJIndex, float bias) {
    // Get Layers I and neuronJ
    Layer layerI = getLayer(layerIIndex);
    Layer layerJ = getLayer(layerJIndex);

    // Loop though all Neurons in layer i
    for (int i = 0; i < layerI.getTensorSize(); ++i) {
      Neuron neuronI = layerI.getNeuron(i);
      // Create new array of output synapses in neuron i
      neuronI.setOutputSynapses(new Synapse[layerJ.getTensorSize()]);
      // Loop through all Neurons in layer j
      for (int j = 0; j < layerJ.getTensorSize(); ++j) {
        Neuron neuronJ = layerJ.getNeuron(j);
        // Create new array of input synapses on neuron j
        // Only if i == 0, to only create the array once per neuron
        if (i == 0) {
          neuronJ.setInputSynapses(new Synapse[layerI.getTensorSize()]);
        }
        // Create and link synapse
        Synapse s = new Synapse(neuronI, bias, neuronJ);
        neuronI.setOutputSynapse(s, j);
        neuronJ.setInputSynapse(s, i);
      }
    }
  }

  public void clear() {
    for (Layer layer : layers) {
      float[] newTensor = new float[layer.getTensorSize()];
      for (int i = 0; i < newTensor.length; ++i) {
        newTensor[i] = 0;
      }
      layer.setTensor(newTensor);
    }
  }

  public Layer[] getLayers() {
    return this.layers;
  }

  public Layer getInputLayer() {
    return getLayers()[0];
  }

  public Layer getOutputLayer() {
    return getLayers()[getLayerCount()-1];
  }

  public int getLayerCount() {
    return this.layers.length;
  }

  public Layer getLayer(int index) {
    if (index >= this.layers.length) {
      throw new RuntimeException(String.format("Cannot get Layer with index '%d', as '%d' is the maximum index", index, this.layers.length-1));
    }
    return this.layers[index];
  }
}
