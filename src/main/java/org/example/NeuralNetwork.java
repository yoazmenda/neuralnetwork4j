package org.example;

import java.io.Serializable;
import java.util.Random;

public class NeuralNetwork implements Serializable {

    private double[] inputLayer;
    private double[][] hiddenLayer;
    private double[] outputLayer;
    private double[][] weightsIH;
    private double[][] weightsHO;
    private double[] biasH;
    private double[] biasO;
    private double learningRate;

    /**
     * Initializes a new neural network with the specified input size, hidden size,
     * output size, learning rate, and hidden layer shape (N, M). Initializes the weights and biases with
     * random values between -1 and 1.
     */
    public NeuralNetwork(int inputSize, int hiddenSize, int hiddenSize2, int outputSize, double learningRate) {
        this.inputLayer = new double[inputSize];
        this.hiddenLayer = new double[hiddenSize][hiddenSize2];
        this.outputLayer = new double[outputSize];

        this.weightsIH = new double[inputSize][hiddenSize2];
        this.weightsHO = new double[hiddenSize2][outputSize];
        Random random = new Random();
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize2; j++) {
                this.weightsIH[i][j] = random.nextDouble() * 2 - 1;
            }
        }
        for (int i = 0; i < hiddenSize2; i++) {
            for (int j = 0; j < outputSize; j++) {
                this.weightsHO[i][j] = random.nextDouble() * 2 - 1;
            }
        }

        this.biasH = new double[hiddenSize2];
        this.biasO = new double[outputSize];
        for (int i = 0; i < hiddenSize2; i++) {
            this.biasH[i] = random.nextDouble() * 2 - 1;
        }
        for (int i = 0; i < outputSize; i++) {
            this.biasO[i] = random.nextDouble() * 2 - 1;
        }

        this.learningRate = learningRate;
    }

    public void train(double[] inputs, double[] targets) {
        // Feed the inputs forward through the network and store the outputs
        double[] outputs = feedForward(inputs);

        // Calculate the errors in the output layer and hidden layer
        double[] outputErrors = new double[outputLayer.length];
        for (int i = 0; i < outputLayer.length; i++) {
            outputErrors[i] = (targets[i] - outputs[i]) * sigmoidDerivative(outputLayer[i]);
        }

        double[] hiddenErrors = new double[hiddenLayer[0].length];
        for (int i = 0; i < hiddenLayer[0].length; i++) {
            double sum = 0.0;
            for (int j = 0; j < outputLayer.length; j++) {
                sum += outputErrors[j] * weightsHO[i][j];
            }
            hiddenErrors[i] = sum * sigmoidDerivative(hiddenLayer[0][i]);
        }

        // Calculate the average loss for this batch
        double loss = 0.0;
        for (int i = 0; i < outputLayer.length; i++) {
            loss += Math.pow(targets[i] - outputs[i], 2);
        }
        loss /= outputLayer.length;

        // Adjust the weights and biases using the errors and the learning rate
        for (int i = 0; i < outputLayer.length; i++) {
            for (int j = 0; j < hiddenLayer[0].length; j++) {
                double delta = learningRate * outputErrors[i] * hiddenLayer[0][j];
                weightsHO[j][i] += delta;
            }
            biasO[i] += learningRate * outputErrors[i];
        }

        for (int i = 0; i < hiddenLayer[0].length; i++) {
            for (int j = 0; j < inputLayer.length; j++) {
                double delta = learningRate * hiddenErrors[i] * inputLayer[j];
                weightsIH[j][i] += delta;
            }
            biasH[i] += learningRate * hiddenErrors[i];
        }
    }


    /**
     * Calculates the outputs of the neural network for the given inputs.
     * Returns an array of the output values.
     */
    public double[] feedForward(double[] inputs) {
        // Set the input layer to the inputs
        for (int i = 0; i < inputLayer.length; i++) {
            inputLayer[i] = inputs[i];
        }

        // Calculate the hidden layer values using the input layer and weightsIH
        for (int i = 0; i < hiddenLayer.length; i++) {
            for (int j = 0; j < hiddenLayer[i].length; j++) {
                double sum = 0.0;
                for (int k = 0; k < inputLayer.length; k++) {
                    sum += inputLayer[k] * weightsIH[k][j];
                }
                sum += biasH[j];
                hiddenLayer[i][j] = sigmoid(sum);
            }
        }

        // Calculate the output layer values using the hidden layer and weightsHO
        for (int i = 0; i < outputLayer.length; i++) {
            double sum = 0.0;
            for (int j = 0; j < hiddenLayer.length; j++) {
                double hiddenSum = 0.0;
                for (int k = 0; k < hiddenLayer[j].length; k++) {
                    hiddenSum += hiddenLayer[j][k] * weightsHO[k][i];
                }
                sum += hiddenSum;
            }
            sum += biasO[i];
            outputLayer[i] = sigmoid(sum);
        }

        // Return the output layer values
        return outputLayer;
    }

    /**
     * Applies the sigmoid activation function to the given value and returns the result.
     */
    public double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    /**
     * Calculates the derivative of the sigmoid function with respect to the given value
     * and returns the result.
     */
    public double sigmoidDerivative(double x) {
        double sigmoidX = sigmoid(x);
        return sigmoidX * (1 - sigmoidX);
    }
}
