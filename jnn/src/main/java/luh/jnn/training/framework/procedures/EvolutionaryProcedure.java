package luh.jnn.training.framework.procedures;

import java.util.Random;

import luh.jnn.Logging;
import luh.jnn.nn.Layer;
import luh.jnn.nn.NNEvaluator;
import luh.jnn.nn.NeuralNetwork;
import luh.jnn.nn.Neuron;
import luh.jnn.nn.Synapse;
import luh.jnn.training.TrainingData;
import luh.jnn.training.framework.TrainingConfig;

public class EvolutionaryProcedure implements TrainingProcedure {
  private int instances;
  private float learningRate;
  private NeuralNetwork nn;
  private NeuralNetwork best;
  private float[] bestTensor;
  private float bestError;
  private TrainingConfig config;
  private NNEvaluator evaluator;
  private TrainingData data;
  private Random rnd;

  public EvolutionaryProcedure(int instances, float learningRate) {
    this.instances = instances;
    this.learningRate = learningRate;
  }

  private NeuralNetwork introduceNoise() {
    NeuralNetwork current = nn.clone();


    for (int i = 0; i < current.getLayerCount(); i++) {
      Layer currentLayer = current.getLayer(i);
      currentLayer.setBias(currentLayer.getBias()+rnd.nextFloat(-this.learningRate/2f, this.learningRate/2f));

      for (int j = 0; j < currentLayer.getTensorSize(); j++) {
        Neuron currentNeuron = currentLayer.getNeuron(j);
        if (currentNeuron.getOutputSynapses() == null) break;
        for (int k = 0; k < currentNeuron.getOutputSynapses().length; k++) {
          Synapse currentSynapse = currentNeuron.getOutputSynapses()[k];
          currentSynapse.setWeight(currentSynapse.getWeight()+rnd.nextFloat(-this.learningRate/2f, this.learningRate/2f));
        }
      }
    }

    return current;
  }

  private float calculateError(float[] tensor) {
    float error = 0.0f;

    for (int i = 0; i < this.data.getExpectedOutput().length; i++) {
      error += Math.abs(this.data.getExpectedOutput()[i]-tensor[i]);
    }

    error /= this.data.getExpectedOutput().length;

    return error;
  }

	@Override
	public NeuralNetwork train(NeuralNetwork nn, TrainingConfig config) {
    this.nn = nn;
    this.config = config;
    this.rnd = new Random();
    this.best = nn.clone();
    this.nn.clear();

    this.data = this.config.getTrainingData().getNextData();
    
    this.evaluator = new NNEvaluator(nn);
    this.evaluator.setConditioning(data.getInput());
    this.evaluator.fullEvaluation();
    this.bestTensor = this.evaluator.getResult();
    this.nn.clear();

    this.bestError = calculateError(this.bestTensor);

    for (int i = 0; i < this.instances; i++) {
      NeuralNetwork current = introduceNoise();
      this.evaluator = new NNEvaluator(current);
      this.evaluator.setConditioning(this.data.getInput());
      this.evaluator.fullEvaluation();
      float[] currentTensor = this.evaluator.getResult();

      float error = calculateError(currentTensor);
      
      // Logging.logger.info(String.format("Accuracy: %f (error: %f)", 1.0f-error, error));

      if (error < this.bestError) {
        Logging.logger.info(String.format("Found model with higher accuracy (Accuracy: %f -> Error: %f)",
              1.0f-error,
              error));
        this.best = current;
        this.bestTensor = currentTensor;
        this.bestError = error;
        this.best.clear();
      }
    }

    return this.best;
	}
}
