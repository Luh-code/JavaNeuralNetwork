package luh.jnn.training.framework.procedures;

import luh.jnn.Logging;
import luh.jnn.nn.*;
import luh.jnn.training.TrainingData;
import luh.jnn.training.framework.TrainingConfig;
import luh.jnn.training.framework.procedures.configuration.EvolutionaryConfiguration;
import luh.jnn.training.framework.procedures.configuration.ProcedureConfiguration;

import java.util.Random;

public class EvolutionaryProcedureOld implements TrainingProcedure {
  private NeuralNetwork nn;
  private NeuralNetwork best;
  private float[] bestTensor;
  private float bestError;
  private TrainingConfig config;
  private NNEvaluator evaluator;
  private TrainingData data;
  private Random rnd;

  private EvolutionaryConfiguration procCon;

  public EvolutionaryProcedureOld() {
  }

  private NeuralNetwork introduceNoise() {
    NeuralNetwork current = nn.clone();


    for (int i = 0; i < current.getLayerCount(); i++) {
      Layer currentLayer = current.getLayer(i);
      currentLayer.setBias(currentLayer.getBias()+rnd.nextFloat(
        -this.procCon.getLearningRate()/2f, this.procCon.getLearningRate()/2f));

      for (int j = 0; j < currentLayer.getTensorSize(); j++) {
        Neuron currentNeuron = currentLayer.getNeuron(j);
        if (currentNeuron.getOutputSynapses() == null) break;
        for (int k = 0; k < currentNeuron.getOutputSynapses().length; k++) {
          Synapse currentSynapse = currentNeuron.getOutputSynapses()[k];
          currentSynapse.setWeight(currentSynapse.getWeight()+rnd.nextFloat(
            -this.procCon.getLearningRate()/2f, this.procCon.getLearningRate()/2f));
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
  public boolean compatibleConfiguration(ProcedureConfiguration procCon) {
    return procCon instanceof EvolutionaryConfiguration;
  }

  @Override
  public void init(ProcedureConfiguration procCon) {
    if (!compatibleConfiguration(procCon)) {
      Logging.logger.fatal("EvolutionaryProcedure needs to be initialized with a EvolutionaryConfiguration");
      System.exit(1);
    }

    this.procCon = (EvolutionaryConfiguration) procCon;
  }

  @Override
  public NeuralNetwork train(NeuralNetwork nn, TrainingConfig config) {
    this.nn = nn;
    this.config = config;
    this.rnd = new Random();
    this.best = nn.clone();
    this.nn.clear();

    this.data = this.config.getTrainingDataSet().getNextData();

    this.evaluator = new NNEvaluator(nn);
    this.evaluator.setConditioning(data.getInput());
    this.evaluator.fullEvaluation();
    this.bestTensor = this.evaluator.getResult();
    this.nn.clear();

    this.bestError = calculateError(this.bestTensor);

    for (int i = 0; i < this.procCon.getInstances(); i++) {
      NeuralNetwork current = introduceNoise();
      this.evaluator = new NNEvaluator(current);
      this.evaluator.setConditioning(this.data.getInput());
      this.evaluator.fullEvaluation();
      float[] currentTensor = this.evaluator.getResult();

      float error = calculateError(currentTensor);

      // Logging.logger.info(String.format("Accuracy: %f (error: %f)", 1.0f-error, error));

      if (error < this.bestError) {
        Logging.logger.info(String.format("Found model with higher accuracy (Accuracy: %f.6/%f.4%% -> Error: %f.6/%f.4%%",
          1.0f-error, (1.0f-error)*100.0f,
          error, error*100.0f));
        this.best = current;
        this.bestTensor = currentTensor;
        this.bestError = error;
        this.best.clear();
      }
    }

    return this.best;
  }
}