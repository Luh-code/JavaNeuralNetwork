package luh.jnn.serialization;

import java.io.File;

import luh.jnn.Logging;
import luh.jnn.nn.NeuralNetwork;

public class NNLoader {
  private NeuralNetworkDeserializer deserializer;

  public NNLoader(NeuralNetworkDeserializer deserializer) {
    this.deserializer = deserializer;
  }

  public NeuralNetwork loadFromFile(File file) {
    if (!file.exists()) {
      Logging.logger.error(String.format("The file '%s' doesn't exist", file.getPath()));
      return null;
    }
    
    NeuralNetwork temp = this.deserializer.deserialize(file);
    if (temp != null) {
      Logging.logger.info(String.format("Successfully deserialized neural network from '%s'", file.getPath()));
    }
    return temp;
  }
}
