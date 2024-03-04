package luh.jnn.training.framework.procedures.configuration;

import luh.jnn.nn.CostFunction;
import luh.jnn.training.framework.optimizers.Optimizer;

import java.util.function.BiFunction;
import java.util.function.Function;

public class BackpropagationConfiguration implements ProcedureConfiguration {
	private CostFunction cost;


	private Optimizer optimizer;

	public BackpropagationConfiguration(CostFunction cost, Optimizer optimizer) {
		this.cost = cost;
		this.optimizer = optimizer;
	}

	public CostFunction getCost() {
		return cost;
	}

	public void setCost(CostFunction cost) {
		this.cost = cost;
	}

	public Optimizer getOptimizer() {
		return optimizer;
	}

	public void setOptimizer(Optimizer optimizer) {
		this.optimizer = optimizer;
	}
}
