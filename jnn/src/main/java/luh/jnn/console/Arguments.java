package luh.jnn.console;

import java.util.ArrayList;

import luh.jnn.console.arghandlers.ArgumentHandler;
import luh.jnn.console.arghandlers.VerbosityHandler;

public class Arguments {
  private String[] args; 
  
  private int currentArgument;

  private static ArrayList<ArgumentHandler> registeredHandlers = new ArrayList<>();

  public static <T extends ArgumentHandler> void registerArgumentHandler(T handler) {
    if (containsHandler(handler.getClass())) {
      return;
    }
    registeredHandlers.add(handler);
  }

  private static boolean containsHandler(Class<? extends ArgumentHandler> clazz) {
    for (ArgumentHandler h : registeredHandlers) {
      if (clazz.isInstance(h)) {
        return true;
      }
    }
    return false;
  }

  public Arguments(String[] args){
    this.args = args;
    this.currentArgument = 0;
    VerbosityHandler.addHandler();
  }

  public void processArguments() {
    while (currentArgument < args.length) {
      String argument = popArgument();

      for (ArgumentHandler handler : registeredHandlers) {
        if (!handler.checkKeys(argument)) {
          continue;
        }

        handler.process(this);
        break;
      }
    }
  }

  public String popArgument() {
    currentArgument++;
    return args[currentArgument-1];
  }
}
