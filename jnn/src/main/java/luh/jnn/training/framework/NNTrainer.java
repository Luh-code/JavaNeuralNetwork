package luh.jnn.training.framework;

import java.io.File;
import java.io.IOError;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import luh.jnn.Logging;
import luh.jnn.nn.NeuralNetwork;
import luh.jnn.serialization.NNSaver;
import luh.jnn.training.framework.procedures.TrainingProcedure;

public class NNTrainer {
  private TrainingProcedure proc;
  private NeuralNetwork nn;
  private NNSaver saver;
  private TrainingConfig config;
  private int currentIteration = 0;

  public NNTrainer(TrainingProcedure proc, NeuralNetwork nn, NNSaver saver) {
    this.proc = proc;
    this.nn = nn;
    this.saver = saver;
  }
  
  private void createCheckpointLocation() {
    Path directoryPath = Paths.get(this.config.getCheckpointLocation().getPath());

    if (!this.config.getCheckpointLocation().exists()) {
      try {
        Files.createDirectories(this.config.getCheckpointLocation().toPath());
      } catch (IOException e) {
        Logging.logger.error(String.format("Could not create all folders leading up to and including '%s'", this.config.getCheckpointLocation().getAbsolutePath()));
        Logging.logger.trace(e);
        return;
      }
    }
  }

  private void saveIfDistanceReached() {
    if (currentIteration%this.config.getCheckpointDistance() == 0) {
      createCheckpointLocation();
      int decimalDigitCount = (int)Math.log10(this.config.getIterationCount())+1;
      this.saver.saveToFile(new File(String.format("%s/%s_checkpoint%0"+String.valueOf(decimalDigitCount)+"d.bin",
              this.config.getCheckpointLocation().getPath(), this.config.getModelName(), currentIteration)), nn);
    }
  }

  private void iteration() {
    this.nn = this.proc.train(this.nn, config);
    saveIfDistanceReached();
  }

  public void train(TrainingConfig config) {
    this.config = config;
    if (this.nn == null) {
      Logging.logger.error("Need a neural network for training");
      return;
    }
    
    if (this.proc == null) {
      Logging.logger.error("Need a training procedure for training");
      return;
    }

    if (this.config.getTrainingData().dataCount() == 0) {
      Logging.logger.error("Need at least one piece of training data");
      return;
    }
    
    while (this.currentIteration < this.config.getIterationCount()) {
      iteration();
      this.currentIteration++;
    }
  }

  public NeuralNetwork getTrainedNeuralNetwork() {
    return this.nn;
  }
}
