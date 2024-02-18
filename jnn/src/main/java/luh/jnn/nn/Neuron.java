package luh.jnn.nn;

public class Neuron {
  private Synapse[] inputSynapses;
  private Synapse[] outputSynapses;
  private float z;
  
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
}
