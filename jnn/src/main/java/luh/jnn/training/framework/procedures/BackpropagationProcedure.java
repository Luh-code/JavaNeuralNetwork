package luh.jnn.training.framework.procedures;

import luh.jnn.Logging;
import luh.jnn.nn.*;
import luh.jnn.training.TrainingData;
import luh.jnn.training.framework.TrainingConfig;
import luh.jnn.training.framework.procedures.configuration.BackpropagationConfiguration;
import luh.jnn.training.framework.procedures.configuration.ProcedureConfiguration;

import java.util.Arrays;

public class BackpropagationProcedure implements TrainingProcedure {
	private NeuralNetwork nn;
	private TrainingConfig config;
	private BackpropagationConfiguration procCon;
	private NNEvaluator evaluator;
	private Float[][][] deltaWeights;
	private int deltaWeightsAdded;
	private Float[][] deltaBiases;
	private int deltaBiasesAdded;

	@Override
	public boolean compatibleConfiguration(ProcedureConfiguration procCon) {
		return procCon instanceof BackpropagationConfiguration;
	}

	@Override
	public void init(ProcedureConfiguration procCon) {
		if (!compatibleConfiguration(procCon)) {
			Logging.logger.fatal("BackpropagationProcedure needs to be initialized with a BackpropagationConfiguration");
			System.exit(1);
		}

		this.procCon = (BackpropagationConfiguration) procCon;
	}

	private float[] floatObjToPrimitive(Float[] arr) {
		float[] res = new float[arr.length];
		for (int i = 0; i < res.length; i++) {
			res[i] = arr[i];
		}
		return res;
	}

	private Float[] floatPrimitiveToObj(float[] arr) {
		Float[] res = new Float[arr.length];
		for (int i = 0; i < res.length; i++) {
			res[i] = arr[i];
		}
		return res;
	}

	@Override
	public NeuralNetwork train(NeuralNetwork nn, TrainingConfig config) {
		this.nn = nn;
		this.config = config;
		this.deltaWeightsAdded = 0;
		this.deltaBiasesAdded = 0;

		TrainingData data = this.config.getTrainingDataSet().getNextData();

		this.evaluator = new NNEvaluator(this.nn);
		this.evaluator.setConditioning(data.getInput());
		this.evaluator.fullEvaluation();
		//this.nn.clear();

		//float[] result = this.evaluator.getResult();

		int layerIndex = this.nn.getLayerCount()-1;

		Layer layer = this.nn.getLayer(layerIndex);

		Float[] dCdO = this.procCon.getCost().getDerivative().apply(
			floatPrimitiveToObj(data.getExpectedOutput()),
			floatPrimitiveToObj(layer.getTensor()));

		this.deltaWeights = new Float[this.nn.getLayerCount()][][];
		this.deltaBiases = new Float[this.nn.getLayerCount()][];

		do {
			//Float[] dCdOActDiff = layer.getActivationFunction().getDerivativeForArray(dCdO);
			Float[] activationDerivative = layer.getActivationFunction().getDerivativeForArray(floatPrimitiveToObj(layer.getTensor()));

			Float[] dCdI = new Float[layer.getTensorSize()];
			// Calculate element product
			for (int i = 0; i < dCdI.length; i++) {
				dCdI[i] = dCdO[i]*activationDerivative[i];
			}

			Float[] precedingLayerTensor =
				floatPrimitiveToObj(this.nn.getLayer(layerIndex-1).getTensor());
			Float[][] dCdW = new Float[precedingLayerTensor.length][dCdO.length];
			// Calculate outer product
			for (int i = 0; i < dCdO.length; i++) {
				for (int j = 0; j < precedingLayerTensor.length; j++) {
					dCdW[j][i] = dCdO[i] * precedingLayerTensor[j];
				}
			}

			if (this.deltaWeights[layerIndex] == null)
				this.deltaWeights[layerIndex] = dCdW.clone();
			else {
				for (int i = 0; i < this.deltaWeights[layerIndex].length; i++) {
					for (int j = 0; j < this.deltaWeights[layerIndex][i].length; j++) {
						this.deltaWeights[layerIndex][i][j] += dCdW[i][j];
					}
				}
			}
			deltaWeightsAdded++;
			if (this.deltaBiases[layerIndex] == null)
				this.deltaBiases[layerIndex] = dCdI.clone();
			else {
				for (int i = 0; i < this.deltaBiases[layerIndex].length; i++) {
					this.deltaBiases[layerIndex][i] += dCdI[i];
				}
			}
			deltaBiasesAdded++;

			dCdO = new Float[layer.getTensorSize()];
			for (int i = 0; i < layer.getTensorSize(); i++) {
				Synapse[] inputSynapses = layer.getNeuron(i).getInputSynapses();
				if (inputSynapses == null)
					continue;
				float[] weights = new float[inputSynapses.length];
				for (int j = 0; j < inputSynapses.length; j++) {
					weights[j] = inputSynapses[i].getWeight();
				}

				//dCdO[i] = 0.0f;
				for (int j = 0; j < dCdI.length; j++) {
					dCdO[i] = weights[j] * dCdI[j];
				}
			}

			layer = this.nn.getLayer(layerIndex--);
		} while (layerIndex-1 < this.nn.getLayerCount()); // maybe not -1 here

		updateFromLearning();

		return this.nn;
	}

	private void updateFromLearning() {
		for (int i = 0; i < this.nn.getLayerCount(); i++) {
			Layer l = this.nn.getLayer(i);
			if (l.getNeuron(0).getInputSynapses() == null || l.getNeuron(0).getInputSynapses().length == 0) {
				continue;
			}
			updateWeights(i);
			updateBiases(i);
		}
	}

	private Float[][] averageOfMatrix(Float[][] mat, int count) {
		Float[][] average = mat.clone();
		float factor = 1.0f / count;
		for (int i = 0; i < average.length; i++) {
			for (int j = 0; j < average[i].length; j++) {
				average[i][j] *= factor;
			}
		}

		return average;
	}

	private void updateWeights(int layerIndex) {
		if (deltaWeightsAdded > 0) {
			Float[][] average_dW = averageOfMatrix(this.deltaWeights[layerIndex], this.deltaWeightsAdded);
			this.procCon.getOptimizer().updateWeights(this.deltaWeights[layerIndex], average_dW);
			Layer layer = this.nn.getLayer(layerIndex);
			for (int i = 0; i < layer.getTensorSize(); i++) {
				Synapse[] outputSynapses = layer.getNeuron(i).getOutputSynapses();
				for (int j = 0; j < outputSynapses.length; j++) {
					outputSynapses[j].setWeight(outputSynapses[j].getWeight()+this.deltaWeights[layerIndex][i][j]);
				}
			}
			this.deltaWeights[layerIndex] = null;
			this.deltaWeightsAdded = 0;
		}
	}

	private Float[] averageVector(Float[] vec, int count) {
		Float[] average = vec.clone();
		float factor = 1.0f / count;
		for (int i = 0; i < average.length; i++) {
			average[i] *= factor;
		}

		return average;
	}

	private void updateBiases(int layerIndex) {
		if (deltaBiasesAdded > 0) {
			Float[] average_bias = averageVector(this.deltaBiases[layerIndex], this.deltaBiasesAdded);
			this.procCon.getOptimizer().updateBiases(this.deltaBiases[layerIndex], average_bias);
			Layer layer = this.nn.getLayer(layerIndex);
			for (int i = 0; i < layer.getTensorSize(); i++) {
				layer.getBiases()[i] += this.deltaBiases[layerIndex][i];
			}
			this.deltaBiases[layerIndex] = null;
			this.deltaBiasesAdded = 0;
		}
	}
}
