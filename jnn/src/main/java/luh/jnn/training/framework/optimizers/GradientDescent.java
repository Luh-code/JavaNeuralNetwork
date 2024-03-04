package luh.jnn.training.framework.optimizers;

public class GradientDescent implements Optimizer{
	private float learningRate;

	public GradientDescent(float learningRate) {
		this.learningRate = learningRate;
	}

	@Override
	public void updateWeights(Float[][] weights, Float[][] dCdW) {
		Float[][] temp = dCdW.clone();
		for (int i = 0; i < temp.length; i++) {
			for (int j = 0; j < temp[i].length; j++) {
				temp[i][j] *= this.learningRate;
			}
		}

		for (int i = 0; i < weights.length; i++) {
			for (int j = 0; j < weights[i].length; j++) {
				weights[j][i] -= temp[j][i];
			}
		}
	}

	@Override
	public void updateBiases(Float[] biases, Float[] dCdB) {
		Float[] temp = dCdB.clone();
		for (int i = 0; i < temp.length; i++) {
			temp[i] *= this.learningRate;
		}

		for (int i = 0; i < biases.length; i++) {
			biases[i] -= temp[i];
		}
	}
}
