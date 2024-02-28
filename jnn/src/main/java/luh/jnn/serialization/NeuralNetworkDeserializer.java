package luh.jnn.serialization;

import java.io.File;

import luh.jnn.nn.NeuralNetwork;

public interface NeuralNetworkDeserializer {

  NeuralNetwork deserialize(File file);
}
