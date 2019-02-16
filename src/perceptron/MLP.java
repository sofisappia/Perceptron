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
	     for(int i =0; i<epochs; i++){
	       activations = this.forward(X);
	       dW = this.backPropagation(activations, X, y);
	       this.updateWeights(dW);
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
	      Vi = Operations.sigmoid(hi, false);
	      if(i != layers.size()-1){
	        Vi = Operations.extend(Vi, 1, true);
	       }
	       activationOutputs.add(Vi);       
	     }
	     return activationOutputs; 
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
	     
	     di = Operations.subtract(y, ai); 
	     di = Mat.dotMultiply(di, Operations.sigmoid(ai,true));// delta of the output layer
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
	}

