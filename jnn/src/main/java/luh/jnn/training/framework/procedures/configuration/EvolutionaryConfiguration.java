package luh.jnn.training.framework.procedures.configuration;

public class EvolutionaryConfiguration implements ProcedureConfiguration {

	private int instances;
	private boolean checkAll;
	private float learningRate;

	public EvolutionaryConfiguration(int instances, boolean checkAll, float learningRate) {
		this.instances = instances;
		this.checkAll = checkAll;
		this.learningRate = learningRate;
	}

	public int getInstances() {
		return instances;
	}

	public void setInstances(int instances) {
		this.instances = instances;
	}

	public boolean isCheckAll() {
		return checkAll;
	}

	public void setCheckAll(boolean checkAll) {
		this.checkAll = checkAll;
	}

	public float getLearningRate() {
		return learningRate;
	}

	public void setLearningRate(float learningRate) {
		this.learningRate = learningRate;
	}
}
