package luh.jnn.nn;

import luh.jnn.Logging;

public class Synapse {
  private Neuron input;
  private float weight;
  private Neuron output;
  
  public Synapse(Neuron input, float weight, Neuron output) {
    this.input = input;
    this.weight = weight;
    this.output = output;
  }

  public void propogateValue() {
    if (input == null || output == null) {
      Logging.logger.fatal("An input neuron, an output neuron and an associated weight is required for signal propogation though a synapse");
      System.exit(1);
    }
    output.setZ(output.getZ() + (input.getZ()*weight));
  }

  public Neuron getInput() {
    return input;
  }
  public void setInput(Neuron input) {
    this.input = input;
  }
  public float getWeight() {
    return weight;
  }
  public void setWeight(float weight) {
    this.weight = weight;
  }
  public Neuron getOutput() {
    return output;
  }
  public void setOutput(Neuron output) {
    this.output = output;
  }


}
