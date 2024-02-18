package luh.jnn;

import org.apache.logging.log4j.core.Logger;

import luh.jnn.console.Arguments;
import luh.jnn.nn.*;
import luh.jnn.console.arghandlers.*;

public class Main {
  
  public static void main(String[] args) {
    Logging.setupLogger();
    
    Arguments a = new Arguments(args);
    a.processArguments();
    
    Logging.logger.debug("debug");
    Logging.logger.info("info");
    Logging.logger.warn("warn");
    Logging.logger.error("error");
    Logging.logger.fatal("fatal");

    Layer[] layers = new Layer[] {
      new Layer(10, 0),
          new Layer(6, 0),
          new Layer(8, 0),
          new Layer(3, 0)
    };
    NeuralNetwork nn = new NeuralNetwork(layers);
  }
}
