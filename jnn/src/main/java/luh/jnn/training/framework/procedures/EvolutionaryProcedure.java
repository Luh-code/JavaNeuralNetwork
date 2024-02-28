package luh.jnn.training.framework.procedures;

import luh.jnn.nn.NeuralNetwork;
import luh.jnn.training.framework.TrainingConfig;

public class EvolutionaryProcedure implements TrainingProcedure {
  private int instances;
  private NeuralNetwork nn; 
  private TrainingConfig config;

  public EvolutionaryProcedure(int instances) {
    this.instances = instances;
  }

	@Override
	public void train(NeuralNetwork nn, TrainingConfig config) {
    this.nn = nn;
    this.config = config;
	}
}
