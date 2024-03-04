package luh.jnn;

import java.io.File;
import java.util.Arrays;

import luh.jnn.console.Arguments;
import luh.jnn.nn.*;
import luh.jnn.serialization.NNLoader;
import luh.jnn.serialization.NNSaver;
import luh.jnn.serialization.NeuralNetworkOISDeserializer;
import luh.jnn.serialization.NeuralNetworkOOSSerializer;
import luh.jnn.training.TrainingData;
import luh.jnn.training.TrainingDataSet;
import luh.jnn.training.framework.NNTrainer;
import luh.jnn.training.framework.TrainingConfig;
import luh.jnn.training.framework.procedures.EvolutionaryProcedure;
import luh.jnn.training.framework.procedures.configuration.EvolutionaryConfiguration;

public class Main {
  
  public static void main(String[] args) {
    Logging.setupLogger();
    
    Arguments a = new Arguments(args);
    a.processArguments();

    Layer[] layers = new Layer[] {
      new Layer(6, 1f),
      new Layer(16, 1f),
      new Layer(16, 1f),
      new Layer(3, 1f)
    };
    NeuralNetwork nn = new NeuralNetwork(layers);
    nn.initalizeDenseNeuralNetwork();
    nn.clear();

    Layer outputLayer = nn.getOutputLayer();
    Neuron firstNeuron = outputLayer.getNeuron(0);
    Synapse firstSynapse = firstNeuron.getInputSynapses()[0];
    firstSynapse.setWeight(-0.7f);
    
    NNEvaluator evaluator = new NNEvaluator(nn);
    float[] conditioning = new float[] {1.0f, 0.84f, 0.68f, 0.49f, 0.33f, 0.16f};
    // float[] conditioning = new float[nn.getInputLayer().getTensorSize()];
    // Random random = new Random();
    // for (int i = 0; i < conditioning.length; i++) {
    //   conditioning[i] = random.nextFloat(-1.0f, 1.0f);
    // }
    evaluator.setConditioning(conditioning);
    evaluator.fullEvaluation();
    Logging.logger.info(Arrays.toString(evaluator.getResult()));

    NNTrainer trainer = new NNTrainer (new EvolutionaryProcedure(), nn, new NNSaver(new NeuralNetworkOOSSerializer()));
    trainer.train(new TrainingConfig (new TrainingDataSet(new TrainingData[] {
      new TrainingData(new float[] {1.0f, 0.84f, 0.68f, 0.49f, 0.33f, 0.16f}, new float[] {1.0f, 0.5f, 0.25f}),
      // new TrainingData(new float[] {0.16f, 0.33f, 0.49f, 0.68f, 0.84f, 1.0f}, new float[] {0.25f, 0.5f, 1.0f}),
    }), 2000, 50, new File("models/test/checkpoints/"), "test"
      , new EvolutionaryConfiguration(200, true, 0.2f)));
    nn = trainer.getTrainedNeuralNetwork();

    NNSaver saver = new NNSaver(new NeuralNetworkOOSSerializer());
    saver.saveToFile(new File("test.bin"), nn);
    
    NNLoader loader = new NNLoader(new NeuralNetworkOISDeserializer());
    NeuralNetwork deserializedNN = loader.loadFromFile(new File("test.bin"));

    NNEvaluator evaluator2 = new NNEvaluator(deserializedNN);
    evaluator2.setConditioning(conditioning);
    evaluator2.fullEvaluation();
    Logging.logger.info(Arrays.toString(evaluator2.getResult()));
    conditioning = new float[] {0.16f, 0.33f, 0.49f, 0.68f, 0.84f, 1.0f};
    deserializedNN.clear();
    evaluator2.setConditioning(conditioning);
    evaluator2.fullEvaluation();
    Logging.logger.info(Arrays.toString(evaluator2.getResult()));
  }
}
