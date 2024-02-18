package luh.jnn.nn;

import luh.jnn.Logging;

public class NNEvaluator {
  private NeuralNetwork nn;
  private float[] conditioning;
  private float[] result;

  public NNEvaluator(NeuralNetwork nn) {
    this.nn = nn;
    this.conditioning = null;
    this.result = null;
  }

  public void setConditioning(float[] conditioning) {
    this.conditioning = conditioning;
  }

  public void fullEvaluation() {
    applyConditioning();
    for (int i = 1; i < nn.getLayerCount(); ++i) {
      evalutateLayer(i);
    }
    readOutputTensor();
  }

  private void readOutputTensor() {
    this.result = nn.getLayer(nn.getLayerCount()-1).getTensor();
  }

  private void applyConditioning() {
    if (this.nn == null) {
      Logging.logger.fatal("Neural Network required for applying conditioning");
      System.exit(1);
    }

    if (conditioning == null) {
      Logging.logger.fatal("Cannot apply empty conditioning to Neural Network");
      System.exit(1);
    }

    Layer input = nn.getLayer(0);
    if (input.getTensorSize() != conditioning.length) {
      Logging.logger.error(String.format("Tensor size mismatch '%d' to '%d'"));
    }

    input.setTensor(conditioning);
  }

  private void evalutateLayer(int layerIndex) {
    Layer layer = nn.getLayer(layerIndex);
    for (Neuron n : layer.getNeurons()) {
      for (Synapse s : n.getInputSynapses()) {
        s.propogateValue();
      }
    }
  }

  public NeuralNetwork getNN() {
    return nn;
  }

  public float[] getConditioning() {
    return conditioning;
  }

  public float[] getResult() {
    return result;
  }


}
