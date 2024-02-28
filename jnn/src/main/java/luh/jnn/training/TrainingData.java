package luh.jnn.training;

public class TrainingData {
  private float[] input;
  private float[] expectedOutput;
  
  public TrainingData(float[] input, float[] expectedOutput) {
    this.input = input;
    this.expectedOutput = expectedOutput;
  }

  public float[] getInput() {
    return input;
  }
  public void setInput(float[] input) {
    this.input = input;
  }
  public float[] getExpectedOutput() {
    return expectedOutput;
  }
  public void setExpectedOutput(float[] expectedOutput) {
    this.expectedOutput = expectedOutput;
  }
}
