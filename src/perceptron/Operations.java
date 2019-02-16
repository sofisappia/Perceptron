package perceptron;
import processing.core.PApplet;
import papaya.*;

public class Operations {
	
	private static PApplet sketch;
	
	Operations(PApplet sketch){
		this.sketch = sketch;
	}
	
	public static float[][] dotProduct(float[][] A, float[][] B, boolean columns){ 
	    // columns: boolean to know if data is arranged in columns or not
	    int ARows = A.length; // m1 rows length
	    int BRows = B.length;    // m2 rows length
	    int ACols = A[0].length;
	    int BCols = A[0].length;

	    if((ACols != BCols) || (ARows != BRows)){
	        sketch.println("Error de tamaño");
	        return null; // matrix multiplication is not possible
	    }else{
	      if(columns){
	        float[][] D = new float[ARows][1];
	        for(int i=0; i<ARows; i++){
	          D[i][0] = dotProduct(A[i], B[i]);
	        } return D;
	      }else{
	        float[][] D = new float[1][ACols];
	        for(int i=0; i<ACols; i++){
	          D[0][i] = dotProduct(Mat.transpose(A)[i], Mat.transpose(B)[i]);
	        }  return D;          
	        }
	    }
	}


	public static float dotProduct(float[] A, float[] B){ 
	    // columns: boolean to know if data is arranged in columns or not
	    int Alen = A.length; // m1 rows length
	    int Blen = B.length;    // m2 rows length
	    if(Alen != Blen){
	        sketch.println("Error de tamaño");
	        return 0; 
	    }else{
	      float D = 0;
	        for(int i=0; i<Alen; i++){
	            D += A[i] * B[i];
	        }
	        return D;          
	    }  
	}

	public static float[][] sigmoid(float[][] z, boolean derv){
	// Función que calcula la función de activación sigmoidea
	// y su derivada
	float[][] y = new float[z.length][z[0].length];
	if(derv==true){ // cálculo de la derivada
	    float[][] sigm = sigmoid(z, false);
	    y = Mat.dotMultiply(sigm, subtract(1, sigm));
	 // }
	}
	else{ // cálculo de la función sigmoidea
	  for(int i=0; i<z.length; i++){
	    for(int j=0; j<z[0].length; j++){      
	      y[i][j] = 1 / (1 + sketch.exp(-z[i][j]));
	    }
	  }
	}
	return y;
	}

	public static float[][] subtract(float[][] A, float[][] B){
		int rowsA = A.length;
		int colsA = A[0].length;
		int rowsB = B.length;
		int colsB = B[0].length;
		float[][] C = new float[rowsA][colsA];
	
		if((rowsA != rowsB) || (colsA != colsB)){
		  sketch.println("Dimensiones no coinciden");
		  return null;
		}else{
		  for(int i=0; i<rowsA; i++){
		    for(int j=0; j<colsA; j++){
		      C[i][j] = A[i][j] - B[i][j];  
		    }
		  }
		  return C;
		}
	}

	public static float[][] subtract(float a, float[][] B){
		int rowsB = B.length;
		int colsB = B[0].length;
		float[][] C = new float[rowsB][colsB];
		for(int i=0; i<rowsB; i++){
		  for(int j=0; j<colsB; j++){
		      C[i][j] = a - B[i][j];  
		    }
		 }
		 return C;
	}

	public static float[][] scalarMultiply(float a, float[][] A){
		int rowsA = A.length;
		int colsA = A[0].length;
		float[][] B = new float[rowsA][colsA];
		for(int i=0; i<rowsA; i++){
		  for(int j=0; j<colsA; j++){
		    B[i][j] = a * A[i][j];
		  }
		}
		return B;
	}
	
	public static float[] scalarMultiply(float a, float[] A){
		  int lenA = A.length;
		  float[] B = new float[lenA];
		  for(int i=0; i<lenA; i++){
		      B[i] = a * A[i];
		  }
		  return B;
		}
	public static float[][] extend(float[][] A, float a, boolean columns){
		int rowsA = A.length;
		int colsA = A[0].length;
		float[][] B;
		if(columns){
		  B = new float[rowsA][colsA+1];
		  for(int i=0; i<rowsA; i++){
		    for(int j=0; j<colsA; j++){
		      B[i][j] = A[i][j];
		    }
		    B[i][colsA] = a;
		  }
		}else{
		  B = new float[rowsA+1][colsA];
		  for(int i=0; i<rowsA; i++){
		    for(int j=0; j<colsA; j++){
		      B[i][j] = A[i][j];
		      B[rowsA][j] = a;
		    }
		    
		  }
		}    
		return B;
	}

	public static void shape(float[][] A){
		int rowsA = A.length;
		int colsA = A[0].length;
		sketch.println(rowsA, "x", colsA);
	}

	public static void shape(float[] A){
		int rowsA = A.length;
		sketch.println(rowsA, "x0");
	}	
	
	public static int[] argmax(float[][] y){
		  int rows = y.length;
		  int cols = y[0].length;
		  int[] y_cat = new int[rows];
		  for(int i=0; i<rows; i++){
		    for(int j=0; j<cols; j++){
		      if(y[i][j] == 1){
		        y_cat[i] = j;
		      }
		    }
		  }
		  return y_cat;
		}	

}
