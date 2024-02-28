package luh.jnn.training.framework.procedures;

import luh.jnn.nn.NeuralNetwork;
import luh.jnn.training.framework.TrainingConfig;

public interface TrainingProcedure {
  void train(NeuralNetwork nn, TrainingConfig config);
}

