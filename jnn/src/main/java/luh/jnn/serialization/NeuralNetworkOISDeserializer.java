package luh.jnn.serialization;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;

import luh.jnn.Logging;
import luh.jnn.nn.NeuralNetwork;

public class NeuralNetworkOISDeserializer implements NeuralNetworkDeserializer {

	@Override
	public NeuralNetwork deserialize(File file) {
    try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file.getPath()))) {
      return (NeuralNetwork) ois.readObject();
    } catch (IOException | ClassNotFoundException e) {
      Logging.logger.error(String.format("Could not create ObjectInputStream for '%s'", file.getPath()));
      Logging.logger.trace(e);
      return null;
    }
	}
}
