package luh.jnn.serialization;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.OpenOption;
import java.nio.file.StandardOpenOption;

import luh.jnn.Logging;
import luh.jnn.nn.NeuralNetwork;

public class NNSaver {
  private NeuralNetworkSerializer serializer;

  public NNSaver(NeuralNetworkSerializer serializer) {
    this.serializer = serializer;
  }

  private File createTempFile(File file) {
    // create temp
    File tempFile = new File(file.getPath()+".tmp");
    if (!tempFile.exists()) {
      try {
        tempFile.createNewFile();
      } catch (IOException e) {
        Logging.logger.error("Could not create temp file");
        Logging.logger.trace(e);
        return null;
      }
      Logging.logger.info("Truncating old temp file");
    }

    // copy temp to file to temp 
    try (InputStream in = Files.newInputStream(file.toPath(), (OpenOption)StandardOpenOption.READ)) {
      try (OutputStream out = Files.newOutputStream(tempFile.toPath(), (OpenOption)StandardOpenOption.TRUNCATE_EXISTING)) {
        in.transferTo(out);
      } catch (IOException e) {
        Logging.logger.error(String.format("Cound not open '%s' for writing", tempFile.toPath()));
        Logging.logger.trace(e);
        return null;
      }
    } catch (IOException e) {
      Logging.logger.error(String.format("Could not open '%s' for reading", file.toPath()));
      Logging.logger.trace(e);
      return null;
    }
    return tempFile;
  }

  private boolean transferToBackup(File file) {
    File oldFile = new File(file.getPath());
    File newFile = new File(file.getPath()+".bak");
    File tempFile = null;
    if (newFile.exists()) {
      tempFile = createTempFile(newFile);
      if (tempFile == null) {
        return false;
      }
      newFile.delete();
    }

    if (oldFile.renameTo(newFile)) {
      Logging.logger.info(String.format("Made backup of '%s' at '%s'", file.toPath(), newFile.getAbsoluteFile()));
    }
    else {
      Logging.logger.error(String.format("Failed to make backup of '%s'", file.getPath()));
      newFile.delete();
      if (tempFile != null) {
        tempFile.renameTo(newFile);
      }
      return false;
    }
    if (tempFile != null) {
      tempFile.delete();
    }
    return true;
  }

  public boolean saveToFile(File file, NeuralNetwork nn) {
    if (file.isDirectory()) {
      Logging.logger.error(String.format("The path '%s' does not lead to a file", file.toPath()));
      return false;
    }

    if (file.exists()) {
      if(!transferToBackup(file)) return false;
    }

    if (this.serializer == null) {
      Logging.logger.error("Serializer cannot be null");
      return false;
    }

    serializer.serialize(file, nn);
    
    Logging.logger.info(String.format("Successfully serialized neural network to '%s' (size: %d bytes, path: '%s')", file.getPath(), file.length(), file.getAbsoluteFile()));

    return true;
  }
}
