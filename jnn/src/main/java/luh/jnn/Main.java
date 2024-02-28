package luh.jnn;

import java.io.File;
import java.util.Arrays;
import java.util.Random;

import luh.jnn.console.Arguments;
import luh.jnn.nn.*;
import luh.jnn.serialization.NNSaver;
import luh.jnn.serialization.NeuralNetworkOOSSerializer;

public class Main {
  
  public static void main(String[] args) {
    Logging.setupLogger();
    
    Arguments a = new Arguments(args);
    a.processArguments();

    Layer[] layers = new Layer[] {
      new Layer(10, 1f),
      new Layer(6, 1f),
      new Layer(8, 1f),
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
    float[] conditioning = new float[nn.getInputLayer().getTensorSize()];
    Random random = new Random();
    for (int i = 0; i < conditioning.length; i++) {
      conditioning[i] = random.nextFloat(-1.0f, 1.0f);
    }
    evaluator.setConditioning(conditioning);
    evaluator.fullEvaluation();
    Logging.logger.info(Arrays.toString(evaluator.getResult()));

    NNSaver saver = new NNSaver(new NeuralNetworkOOSSerializer());
    saver.saveToFile(new File("test.bin"), nn);
  }
}
