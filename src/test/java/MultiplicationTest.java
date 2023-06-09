

import org.example.NeuralNetwork;
import org.testng.annotations.Test;

import java.io.*;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class MultiplicationTest {


    @Test
    public void testMultiplication() {
        int datasetSize = 1000;
        int testSize = (int) (datasetSize * 0.2);
        int trainSize = datasetSize - testSize;

        List<double[]> inputs = new ArrayList<>();
        List<double[]> targets = new ArrayList<>();

        // Generate synthetic dataset
        Random random = new Random();
        for (int i = 0; i < datasetSize; i++) {
            double num1 = random.nextDouble();
            double num2 = random.nextDouble();
            double product = num1 * num2;

            inputs.add(new double[]{num1, num2});
            targets.add(new double[]{product});
        }

        // Split the dataset into train and test sets
        List<double[]> trainInputs = inputs.subList(0, trainSize);
        List<double[]> trainTargets = targets.subList(0, trainSize);
        List<double[]> testInputs = inputs.subList(trainSize, datasetSize);
        List<double[]> testTargets = targets.subList(trainSize, datasetSize);

        // Initialize the neural network
        NeuralNetwork nn = new NeuralNetwork(2, 10, 10, 1, 0.4);

        // Train the neural network
        int epochs = 5000;
        double[][] trainInputsArray = trainInputs.toArray(new double[trainSize][2]);
        double[][] trainTargetsArray = trainTargets.toArray(new double[trainSize][1]);
        nn.train(trainInputsArray, trainTargetsArray, epochs);


        // Test the neural network
        int correct = 0;
        double threshold = 0.001; // You can adjust the threshold based on your desired accuracy
        for (int i = 0; i < testSize; i++) {
            double[] output = nn.feedForward(testInputs.get(i));
            double predictedProduct = output[0];
            double actualProduct = testTargets.get(i)[0];

            if (Math.abs(predictedProduct - actualProduct) < threshold) {
                correct++;
            }
        }

        double accuracy = (double) correct / testSize * 100;
        System.out.println("Test accuracy: " + accuracy + "%");


        // Save the trained neural network to a file
        String modelPath = Paths.get("src/main/resources/neural_network_model.ser").toString();
        try (FileOutputStream fos = new FileOutputStream(modelPath);
             ObjectOutputStream oos = new ObjectOutputStream(fos)) {
            oos.writeObject(nn);
            System.out.println("Neural network model saved to: " + modelPath);
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Load the neural network model from a file
        NeuralNetwork loadedNN;
        try (FileInputStream fis = new FileInputStream(modelPath);
             ObjectInputStream ois = new ObjectInputStream(fis)) {
            loadedNN = (NeuralNetwork) ois.readObject();
            System.out.println("Neural network model loaded from: " + modelPath);
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
            return;
        }

        // Example usage of the trained neural network
        double num1 = 0.13;
        double num2 = 0.3;
        double[] input = {num1, num2};
        double[] output = loadedNN.feedForward(input);
        double predictedProduct = output[0];

        System.out.println("Real value of " + num1 + " * " + num2 + " = " + (num1 * num2));
        System.out.println("Neural network approximation: " + predictedProduct);
    }
}
