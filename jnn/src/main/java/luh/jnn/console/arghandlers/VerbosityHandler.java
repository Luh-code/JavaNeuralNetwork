package luh.jnn.console.arghandlers;

import luh.jnn.Logging;
import luh.jnn.console.Arguments;
import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.core.LoggerContext;
import org.apache.logging.log4j.core.config.Configuration;
import org.apache.logging.log4j.core.config.LoggerConfig;

/**
 * VerbosityHandler
 */
public class VerbosityHandler extends ArgumentHandler {
	private VerbosityHandler() {
		super("-v", "--verbosity");
	}

  public static void addHandler() {
    new VerbosityHandler();
  }

  @Override
  public void process(Arguments args) {
    Level level = parseLogLevel(args.popArgument());
    LoggerContext context = (LoggerContext) LogManager.getContext(false);
    Configuration config = context.getConfiguration();
    LoggerConfig rootConfig = config.getLoggerConfig(LogManager.ROOT_LOGGER_NAME);
    rootConfig.setLevel(level);
  }

  private Level parseLogLevel(String token) {
    switch (token.toUpperCase()) {
      case "DEBUG", "INFO", "WARN", "ERROR", "FATAL":
        return Level.getLevel(token.toUpperCase());

      default:
        Logging.logger.error(String.format("Could not set LogLevel to '%s', because it does not exist", token.toUpperCase()));
        break;
    }
    return Logging.logger.getLevel();
  }
}
