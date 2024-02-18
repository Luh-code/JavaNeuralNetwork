package luh.jnn;

import java.io.File;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.core.LoggerContext;

public class Logging {
  public static final Logger logger = LogManager.getLogger(Logger.class);

  public static void setupLogger() {
    System.out.println("static block logging");
    LoggerContext context = (LoggerContext) LogManager.getContext(false);
    File file = new File("./../log4j2.xml");
    if (!file.exists()) {
      logger.error("Logger config not found! Using standard config");
    }

    context.setConfigLocation(file.toURI());
    logger.info("Logger config loaded");
  }
}
