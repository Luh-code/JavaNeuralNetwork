package luh.jnn.training.framework.procedures;

import luh.jnn.nn.NeuralNetwork;
import luh.jnn.training.framework.TrainingConfig;
import luh.jnn.training.framework.procedures.configuration.ProcedureConfiguration;

public interface TrainingProcedure {
  boolean compatibleConfiguration(ProcedureConfiguration procCon);
  void init(ProcedureConfiguration procCon);
  NeuralNetwork train(NeuralNetwork nn, TrainingConfig config);
}

