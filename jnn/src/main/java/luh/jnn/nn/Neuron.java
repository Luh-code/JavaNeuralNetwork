package luh.jnn.nn;

import java.io.Serializable;

public class Neuron implements Serializable, Cloneable {
  private static final long serialVersionUID = 1L;
  private Synapse[] inputSynapses;
  private Synapse[] outputSynapses;
  transient private float z;
  
  public Neuron(Synapse[] inputSynapses) {
    this.inputSynapses = inputSynapses;
  }

  public Neuron() {

  } 

  public float getZ() {
    return z;
  }

  public void setZ(float z) {
    this.z = z;
  }

	public Synapse[] getInputSynapses() {
		return inputSynapses;
	}

	public void setInputSynapses(Synapse[] inputSynapses) {
		this.inputSynapses = inputSynapses;
	}

  public void setInputSynapse(Synapse s, int i) {
    this.inputSynapses[i] = s;
  }

	public Synapse[] getOutputSynapses() {
		return outputSynapses;
	}

	public void setOutputSynapses(Synapse[] outputSynapses) {
		this.outputSynapses = outputSynapses;
	}

  public void setOutputSynapse(Synapse s, int i) {
    this.outputSynapses[i] = s;
  }

  @Override
  public Neuron clone() {
    Neuron temp = new Neuron();
    temp.setZ(this.z);
    if (this.outputSynapses != null) {
      Synapse[] tempOutputSynapses = new Synapse[this.outputSynapses.length];
      for (int i = 0; i < tempOutputSynapses.length; i++) {
        tempOutputSynapses[i] = this.outputSynapses[i].clone();
        tempOutputSynapses[i].setInput(temp);
      }
      temp.setOutputSynapses(tempOutputSynapses); 
    }
    if (this.inputSynapses != null) {
      temp.setInputSynapses(new Synapse[this.inputSynapses.length]);
    }
    return temp;
  }
}
