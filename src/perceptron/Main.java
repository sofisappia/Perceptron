package perceptron;

import papaya.Mat;

public class Main {

	public static void main(String[] args) {
		
		float[][] X = {{0, 0, 0}, {0, 1, 0}, {1, 0, 0}, {1, 0, 1}, {1, 1, 1}};
		float[][] y = {{1,0,0,0,0}, {0,1,0,0,0}, {0,0,1,0,0}, {0,0,0,1,0}, {0,0,0,0,1}};
		
		MLP mlp = new MLP(3, 30, 1, 5);
		
		mlp.compile((float) (0.01), 100000);
		mlp.train(X, y);
		
		float[][] pred = mlp.predict(X); 
		Mat.print(pred, 2);
		
		

	}

}


