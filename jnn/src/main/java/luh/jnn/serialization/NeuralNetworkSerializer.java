package luh.jnn.serialization;

import java.io.File;

import luh.jnn.nn.NeuralNetwork;

public interface NeuralNetworkSerializer {

  boolean serialize(File file, NeuralNetwork nn);
}
