package luh.jnn.console.arghandlers;

import luh.jnn.console.Arguments;

public abstract class ArgumentHandler {
  private String[] keys;

  public ArgumentHandler(String... keys) {
    this.keys = keys;
    Arguments.registerArgumentHandler(this);
  }

  public String[] getKeys() {
    return this.keys;
  }
  
  public abstract void process(Arguments args);

  public boolean checkKeys(String token) {
    for (String key : keys) {
      if (token.equals(key)) {
        return true;
      }
    }
    return false;
  }
}
