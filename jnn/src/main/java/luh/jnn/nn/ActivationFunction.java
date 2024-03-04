package luh.jnn.nn;

import java.util.function.Function;

public class ActivationFunction {
	private Function<Float, Float> function;
	private Function<Float, Float> derivative;

	public ActivationFunction(Function<Float, Float> function, Function<Float, Float> derivative) {
		this.function = function;
		this.derivative = derivative;
	}

	public Float[] getFunctionForArray(Float[] arr) {
		for (int i = 0; i < arr.length; i++) {
			arr[i] = this.function.apply(arr[i]);
		}
		return arr;
	}
	public Float[] getDerivativeForArray(Float[] arr) {
		for (int i = 0; i < arr.length; i++) {
			arr[i] = this.derivative.apply(arr[i]);
		}
		return arr;
	}

	public Function<Float, Float> getFunction() {
		return function;
	}

	public void setFunction(Function<Float, Float> function) {
		this.function = function;
	}

	public Function<Float, Float> getDerivative() {
		return derivative;
	}

	public void setDerivative(Function<Float, Float> derivative) {
		this.derivative = derivative;
	}

	public static final ActivationFunction ReLU = new ActivationFunction(
		x -> x <= 0.0f ? 0.0f : x,
		x -> x <= 0.0f ? 0.0f : 1.0f
	);

	public static final ActivationFunction Sigmoid = new ActivationFunction(
		ActivationFunction::sigmoid,
		x -> sigmoid(x) * (1.0f - sigmoid(x))
	);

	private static float sigmoid(float x) {
		return 1.0f / (1.0f + (float) Math.exp(-x));
	}
}
