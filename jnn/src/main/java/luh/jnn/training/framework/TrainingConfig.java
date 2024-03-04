package luh.jnn.training.framework;

import java.io.File;

import luh.jnn.training.TrainingDataSet;
import luh.jnn.training.framework.procedures.configuration.ProcedureConfiguration;

public class TrainingConfig {
  private TrainingDataSet trainingData;
  private int iterationCount;
  private int checkpointDistance;
  private File checkpointLocation;
  private String modelName;

  private ProcedureConfiguration procedureConfig;
  
  public TrainingConfig(TrainingDataSet trainingData, int iterationCount,
                        int checkpointDistance, File checkpointLocation, String modelName, ProcedureConfiguration procedureConfig) {
    this.trainingData = trainingData;
    this.iterationCount = iterationCount;
    this.checkpointDistance = checkpointDistance;
    this.checkpointLocation = checkpointLocation;
    this.modelName = modelName;
    this.procedureConfig = procedureConfig;
  }

  public void setTrainingData(TrainingDataSet trainingData) {
    this.trainingData = trainingData;
  }
  public int getIterationCount() {
    return iterationCount;
  }
  public void setIterationCount(int iterationCount) {
    this.iterationCount = iterationCount;
  }
  public int getCheckpointDistance() {
    return checkpointDistance;
  }
  public void setCheckpointDistance(int checkpointDistance) {
    this.checkpointDistance = checkpointDistance;
  }
  public File getCheckpointLocation() {
    return checkpointLocation;
  }
  public void setCheckpointLocation(File checkpointLocation) {
    this.checkpointLocation = checkpointLocation;
  }
  public String getModelName() {
    return modelName;
  }
  public void setModelName(String modelName) {
    this.modelName = modelName;
  }
  public TrainingDataSet getTrainingDataSet() {
    return trainingData;
  }

  public ProcedureConfiguration getProcedureConfig() {
    return procedureConfig;
  }

  public void setProcedureConfig(ProcedureConfiguration procedureConfig) {
    this.procedureConfig = procedureConfig;
  }
}
