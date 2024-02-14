package luh.jnn.nn;

public class Neuron {
    private float bias;
    private float[] weights;
    
    public Neuron(float bias, float[] weights) {
      this.bias = bias; 
      this.weights = weights;
    }

    public Neuron(float bias) {
      this.bias = bias;
      this.weights = null;
    }
    
    public void setBias(float bias) {
      this.bias = bias;
    }

    public void setWeights(float[] weights) {
      this.weights = weights;
    }

    public float getBias() {
      return this.bias;
    }

    public float[] getWeights() {
      return this.weights;
    }
}
