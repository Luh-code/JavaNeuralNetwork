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
import luh.jnn.training.framework.procedures.configuration.EvolutionaryConfiguration;
import luh.jnn.training.framework.procedures.configuration.ProcedureConfiguration;

public class EvolutionaryProcedure implements TrainingProcedure {
  private NeuralNetwork nn;
  private NeuralNetwork best;
  private float[] bestTensor;
  private float bestError;
  private TrainingConfig config;
  private NNEvaluator evaluator;
  private TrainingData data;
  private Random rnd;

  private NeuralNetwork mutation;

  private EvolutionaryConfiguration procCon;

  public EvolutionaryProcedure() {
  }

  @Override
  public boolean compatibleConfiguration(ProcedureConfiguration procCon) {
    return procCon instanceof EvolutionaryConfiguration;
  }

  @Override
  public void init(ProcedureConfiguration procCon) {
    if (!compatibleConfiguration(procCon)) {
      Logging.logger.fatal("EvolutionaryProcedure needs to be initalized with a EvolutionaryConfiguration");
      System.exit(1);
    }

    this.procCon = (EvolutionaryConfiguration) procCon;
  }

  private void introduceNoise() {
    for (int i = 0; i < mutation.getLayerCount(); i++) {
      Layer currentLayer = mutation.getLayer(i);
      currentLayer.setBias(best.getLayer(i).getBias()+rnd.nextFloat(
        -this.procCon.getLearningRate()/2f, this.procCon.getLearningRate()/2f));

      for (int j = 0; j < currentLayer.getTensorSize(); j++) {
        Neuron currentNeuron = currentLayer.getNeuron(j);
        if (currentNeuron.getOutputSynapses() == null) break;
        for (int k = 0; k < currentNeuron.getOutputSynapses().length; k++) {
          Synapse currentSynapse = currentNeuron.getOutputSynapses()[k];
          currentSynapse.setWeight(
            best.getLayer(i).getNeuron(j).getOutputSynapses()[k].getWeight()+rnd.nextFloat(
              -this.procCon.getLearningRate()/2f, this.procCon.getLearningRate()/2f));
        }
      }
    }
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

    this.mutation = nn.clone();

    for (int i = 0; i < this.procCon.getInstances(); i++) {
      //NeuralNetwork current = introduceNoise();
      introduceNoise();
      this.evaluator = new NNEvaluator(mutation);
      this.evaluator.setConditioning(this.data.getInput());
      this.evaluator.fullEvaluation();
      float[] currentTensor = this.evaluator.getResult();

      float error = calculateError(currentTensor);
      
      // Logging.logger.info(String.format("Accuracy: %f (error: %f)", 1.0f-error, error));

      if (error < this.bestError) {
        Logging.logger.info(String.format("Found model with higher accuracy (Accuracy: %.6f/%.4f%% -> Error: %.6f/%.4f%%",
              1.0f-error, (1.0f-error)*100.0f,
              error, error*100.0f));
        this.best = mutation.clone();
        this.bestTensor = currentTensor;
        this.bestError = error;
        this.best.clear();
      }
    }

    return this.best;
	}
}
