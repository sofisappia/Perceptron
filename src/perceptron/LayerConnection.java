package perceptron;
import processing.core.*;
import perceptron.Operations;
import papaya.*;

public class LayerConnection {
	  // Crea las conexiones entre las diferentes capas de la red
	  //double[][] weights;
	  private PApplet sketch;
	  private float[][] weights;
	  private int nPrevious;
	  private int nLayer;
	  
	  //Matrix W;
	  
	  LayerConnection(PApplet sketch, int neuronsInPreviousLayer, int neuronsInLayer, boolean isOutputlayer){
	    this.sketch = sketch;
		this.nPrevious = neuronsInPreviousLayer;
	    this.nLayer = neuronsInLayer;
	    // weights initialization
	    if(isOutputlayer){
	      this.nPrevious = nPrevious+1;
	      this.weights = new float[nLayer][nPrevious];
	      this.initWeights();
	   }else{
	      //this.nLayer = nLayer+1;
	      this.nPrevious = nPrevious+1;
	      this.weights = new float[nLayer][nPrevious];
	      this.initWeights(); 
	      //this.weights = extend(this.weights, 1);
	   }
	  }
	   void initWeights(){
	    sketch.randomSeed(2); // busco arreglo jxi
	    for(int i=0; i < nLayer; i++){ // columnas
	      for(int j=0; j < nPrevious; j++){ // filas
	        this.weights[i][j] = sketch.randomGaussian();
	      }
	    }
	   }
	   float[][] getWeights(){ 
	     return weights; 
	   }
	   
	   void adjustWeights(float[][] dW){
	     this.weights = Mat.sum(this.weights, dW);
	   }

};

