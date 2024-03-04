package luh.jnn.nn;

import java.util.function.BiFunction;

public class CostFunction {
	private BiFunction<Float[], Float[], Float> total;
	private BiFunction<Float[], Float[], Float[]> derivative;

	public CostFunction(BiFunction<Float[], Float[], Float> total, BiFunction<Float[], Float[], Float[]> derivative) {
		this.total = total;
		this.derivative = derivative;
	}

	public BiFunction<Float[], Float[], Float> getTotal() {
		return total;
	}

	public void setTotal(BiFunction<Float[], Float[], Float> total) {
		this.total = total;
	}

	public BiFunction<Float[], Float[], Float[]> getDerivative() {
		return derivative;
	}

	public void setDerivative(BiFunction<Float[], Float[], Float[]> derivative) {
		this.derivative = derivative;
	}

	public static final CostFunction Quadratic = new CostFunction(
		(expected, actual) -> {
			Float[] diff = actual.clone();
			for (int i = 0; i < diff.length; i++) {
				diff[i] -= expected[i];
			}

			float res = 0;
			for (int i = 0; i < diff.length; i++) {
				res += (float) Math.pow(diff[i], 2);
			}

			return res;
		},
		(expected, actual) -> {
			Float[] diff = actual.clone();
			for (int i = 0; i < diff.length; i++) {
				diff[i] -= expected[i];
			}

			for (int i = 0; i < diff.length; i++) {
				diff[i] *= diff[i];
			}

			return diff;
		}
	);
}
