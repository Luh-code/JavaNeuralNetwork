package luh.jnn.serialization;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;

import luh.jnn.Logging;
import luh.jnn.nn.NeuralNetwork;

public class NeuralNetworkOOSSerializer implements NeuralNetworkSerializer {

	@Override
	public boolean serialize(File file, NeuralNetwork nn) {
    try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(file.getPath()))) {
      oos.writeObject(nn);
      oos.flush();
      oos.close();
    } catch (IOException e) {
      Logging.logger.error(String.format("Could not create ObjectOutputStream for file '%s'", file.toPath()));
      return false;
    }
    return true;
	}
}
