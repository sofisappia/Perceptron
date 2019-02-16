package perceptron;
import java.util.ArrayList;
import perceptron.LayerConnection;
import perceptron.Operations;
import processing.core.PApplet;
import papaya.*;

	
public class MLP extends PApplet{
	  private int inputLayerUnits;
	  private int hiddenLayerUnits;
	  private int hiddenLayers;
	  private int outputLayerUnits;
	  private float loss;
	  
	  // learning parameters
	  float learningRate;
	  int epochs;
	  ArrayList<LayerConnection> layers = new ArrayList<LayerConnection>();
	  
	  MLP(int tempInputLayerUnits, int tempHiddenLayerUnits, int tempHiddenLayers, int tempOutputLayerUnits){ 
	    this.inputLayerUnits = tempInputLayerUnits;
	    this.hiddenLayerUnits = tempHiddenLayerUnits;
	    this.hiddenLayers = tempHiddenLayers;
	    this.outputLayerUnits = tempOutputLayerUnits;
	    
	  }
	  
	  void compile(float tempLearningRate, int tempEpochs){
	    learningRate = tempLearningRate;
	    epochs = tempEpochs;
	    
	    // Crea arreglo de las conexiones entre cada capa  
	    layers.add(new LayerConnection(this, inputLayerUnits, hiddenLayerUnits, false)); //1st layer
	    for(int i=1; i<hiddenLayers; i++){ // Hidden layers
	      layers.add(new LayerConnection(this, hiddenLayerUnits, hiddenLayerUnits, false));
	    }
	    // output layer
	    layers.add(new LayerConnection(this, hiddenLayerUnits, outputLayerUnits, true));
	  //  print(layers.size()); // Tamaño de las capas ocultas  
	  }
	  
	   
	   void train(float[][] X, float[][] y){
	     ArrayList<float[][]> activations = new ArrayList<float[][]>(); //array con los outputs de cada capa
	     ArrayList<float[][]> dW; // para la corrección de pesos
	     try {
		     int i = 0;
		     while (i<epochs){
		       activations = this.forward(X);
		       dW = this.backPropagation(activations, X, y);
		       this.updateWeights(dW);
		       println("Loss: ", loss);
		       if(this.loss <= 0.01){
		         break;
		       }
		       i++;         
		       }
		       println("trained in ", i, " epochs with a loss of ", this.loss);
	     }catch (IllegalArgumentException e) {
	    	 println("Input nodes must match with number of features.\nOuput nodes must match with number of clasess.");
	     }
	   }
	   
	   ArrayList<float[][]> forward(float[][] X){
	     float[][] hi; // potenciales postsinápticos temporales
	     float[][] W;
	     float[][] Vi = new float[0][0]; // salida después de aplicar función de activación
	     ArrayList<float[][]> activationOutputs = new ArrayList<float[][]>();
	     
	     for(int i=0; i<layers.size(); i++){
	      if(i == 0){
	        hi = Operations.extend(X, 1, true);
	      }else{
	        hi = activationOutputs.get(activationOutputs.size()-1);
	      }
	      W = layers.get(i).getWeights();
	      hi = Mat.multiply(hi, Mat.transpose(W));
	      if(i != layers.size()-1){
	          Vi = Operations.sigmoid(hi, false);
	          Vi = Operations.extend(Vi, 1, true);
	        }else{
	           Vi = softmax(hi);
	         }
	       activationOutputs.add(Vi);       
	     }
	     return activationOutputs; 
	   }
	   
	   void cross_entropy_loss(float[][] S, float[][] y){
		    int n = y.length; // number of examples
		/*    Mat.print(S,  3);
		    println();
		    Mat.print(y, 0);
		    println();*/
		    float[][] L = Operations.dotProduct(S, y, true);
		    float tempLoss = 0;
		   // Mat.print(L, 3);
		  //  println();
		   // delay(1000);
		    for(int i=0; i<n; i++){
		 //   	println("li0: ", -1*log(L[i][0]));
		      tempLoss += -1 * log(L[i][0]);

		    }
		//      println(tempLoss/n);
		  //    delay(3000);
		      this.loss = tempLoss/ n;
		   
		  }	   
	   
	   ArrayList<float[][]> backPropagation(ArrayList<float[][]> activations, float[][] X, float[][] y){
	     ArrayList<float[][]> deltas = new ArrayList<float[][]>();
	     ArrayList<float[][]> deltaW = new ArrayList<float[][]>();
	     // output layer
	     float[][] di;
	     float[][] dWij;
	     // hidden layers
	     float[][] dj;
	     float[][] dWjk;
	     // layer outputs
	     float[][] ai; // last layer output
	     float[][] aj; // hidden layer output
	     
	     // output layer
	     ai = activations.get(activations.size()-1); // output layer
	     aj = activations.get(activations.size()-2); // last hidden layer
	    
	     //update loss
	     cross_entropy_loss(ai, y);
	     
	     di = Operations.subtract(y, ai); 
	     //di = Mat.dotMultiply(di, Operations.sigmoid(ai,true));// delta of the output layer
	     deltas.add(di);
	    
	     dWij = Operations.scalarMultiply(learningRate, di);     
	     dWij = Mat.multiply(Mat.transpose(di), aj);
	     deltaW.add(dWij);
	     
	     // hidden layers
	       
	     for(int i=layers.size()-1; i>0; i--){
	       // dj = (Wij.T * di) .* f'(aj)
	       dj = Mat.multiply(deltas.get(deltas.size()-1), layers.get(i).getWeights());
	       dj = Mat.dotMultiply(dj, Operations.sigmoid(activations.get(i-1), true));
	       deltas.add(dj);
	       
	       dWjk = Operations.scalarMultiply(learningRate, dj);
	       if(i == 1){         
	         dWjk = Mat.multiply(Mat.transpose(dWjk), Operations.extend(X,1,true));
	         deltaW.add(dWjk);
	       }else{ 
	         dWjk = Mat.multiply(dWjk, Mat.transpose(activations.get(i-2)));
	         deltaW.add(dWjk);
	       }
	     }
	     return deltaW;  
	   }

	   
	   void updateWeights(ArrayList<float[][]> dW){
	     for(int i=0; i<layers.size(); i++){
	       layers.get(i).adjustWeights(dW.get(dW.size()-1-i));
	     }
	   
	   }
	   float[][] predict(float[][] X){
	     ArrayList<float[][]> outputs = new ArrayList<float[][]>();
	     outputs = this.forward(X);
	     return outputs.get(outputs.size()-1); 
	   }
	   
	   float[][] softmax(float[][] A){
		   int n_ex = A.length; // number of examples
		   int n_classes = A[0].length; // number of classes
		   float[][] probs = new float[n_ex][n_classes]; 
		   // variables auxiliares
		   float[] tempProb = new float[n_classes];
		   float probSum = 0;
		   for(int i=0; i<n_ex; i++){
		     float max = max(A[i]); // para que el rango de entradas sea cercano a cero (evitar division por cero)  
		     for(int j=0; j<n_classes; j++){
		       tempProb[j] = exp(A[i][j] - max);
		     }
		     probSum = Mat.sum(tempProb);
		     probs[i] = Operations.scalarMultiply(1/probSum, tempProb);
		   }
		   return probs;
		 }	   
	}

