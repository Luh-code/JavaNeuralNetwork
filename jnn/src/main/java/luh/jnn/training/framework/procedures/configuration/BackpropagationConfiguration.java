package luh.jnn.training.framework.procedures.configuration;

import java.util.function.BiFunction;
import java.util.function.Function;

public class BackpropagationConfiguration implements ProcedureConfiguration {
	private BiFunction<Float[], Float[], Float> cost;
	public static final BiFunction<Float[], Float[], Float> mse = (s, y) -> {
		float res = 0.0f;

		for (int i = 0; i < s.length; i++) {
			res += (float) Math.pow(y[i]-s[i], 2);
		}

		res *= 1.0f/s.length;

		return res;
	};

	public BackpropagationConfiguration(BiFunction<Float[], Float[], Float> cost) {
		this.cost = cost;
	}

	public BiFunction<Float[], Float[], Float> getCost() {
		return cost;
	}

	public void setCost(BiFunction<Float[], Float[], Float> cost) {
		this.cost = cost;
	}
}
