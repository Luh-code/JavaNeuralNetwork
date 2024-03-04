package luh.jnn.training.framework.optimizers;

import luh.jnn.nn.NeuralNetwork;

public interface Optimizer {
	void updateWeights(Float[][] weights, Float[][]dCdW);
	void updateBiases(Float[] biases, Float[] dCdB);
}
