package luh.jnn.training.framework.procedures;

import luh.jnn.Logging;
import luh.jnn.nn.NeuralNetwork;
import luh.jnn.training.framework.TrainingConfig;
import luh.jnn.training.framework.procedures.configuration.BackpropagationConfiguration;
import luh.jnn.training.framework.procedures.configuration.ProcedureConfiguration;

public class BackpropagationProcedure implements TrainingProcedure {
	private NeuralNetwork nn;
	private TrainingConfig config;
	private BackpropagationConfiguration procCon;

	@Override
	public boolean compatibleConfiguration(ProcedureConfiguration procCon) {
		return procCon instanceof BackpropagationConfiguration;
	}

	@Override
	public void init(ProcedureConfiguration procCon) {
		if (!compatibleConfiguration(procCon)) {
			Logging.logger.fatal("BackpropagationProcedure needs to be initalized with a BackpropagationConfiguration");
			System.exit(1);
		}

		this.procCon = (BackpropagationConfiguration) procCon;
	}

	@Override
	public NeuralNetwork train(NeuralNetwork nn, TrainingConfig config) {
		this.nn = nn;
		this.config = config;
		return null;
	}
}
