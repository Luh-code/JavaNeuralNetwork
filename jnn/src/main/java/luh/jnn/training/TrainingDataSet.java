package luh.jnn.training;

public class TrainingDataSet {
  private TrainingData[] trainingData;
  private int i = 0;

  public TrainingDataSet(TrainingData[] trainingData) {
    this.trainingData = trainingData;
  }

  public TrainingData[] getTrainingData() {
    return this.trainingData;
  }

  public TrainingData getNextData() {
    return this.trainingData[(i++)%trainingData.length];
  }

  public int dataCount() {
    return this.trainingData.length;
  }
}
