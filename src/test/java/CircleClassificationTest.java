
import org.example.NeuralNetwork;
import org.testng.annotations.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;


public class CircleClassificationTest {

    @Test
    public void testCircleClassification() {
        int datasetSize = 1000;
        int testSize = (int) (datasetSize * 0.2);
        int trainSize = datasetSize - testSize;

        List<double[]> inputs = new ArrayList<>();
        List<double[]> targets = new ArrayList<>();

        // Generate synthetic dataset
        Random random = new Random();
        double innerRadius = 1.0;
        double outerRadius = 2.0;

        for (int i = 0; i < datasetSize; i++) {
            double angle = 2 * Math.PI * random.nextDouble();
            double distance;
            int targetClass;

            if (random.nextBoolean()) {
                distance = innerRadius * random.nextDouble();
                targetClass = 0;
            } else {
                distance = innerRadius + (outerRadius - innerRadius) * random.nextDouble();
                targetClass = 1;
            }

            double x = distance * Math.cos(angle);
            double y = distance * Math.sin(angle);

            inputs.add(new double[]{x, y});
            targets.add(new double[]{targetClass});
        }

        // Split the dataset into train and test sets
        List<double[]> trainInputs = inputs.subList(0, trainSize);
        List<double[]> trainTargets = targets.subList(0, trainSize);
        List<double[]> testInputs = inputs.subList(trainSize, datasetSize);
        List<double[]> testTargets = targets.subList(trainSize, datasetSize);

        // Initialize the neural network
        NeuralNetwork nn = new NeuralNetwork(2, 200, 1, 0.1);

        // Train the neural network
        int epochs = 5000;
        for (int i = 0; i < epochs; i++) {
            for (int j = 0; j < trainSize; j++) {
                nn.train(trainInputs.get(j), trainTargets.get(j));
            }

            // Print progress
            if (i % 500 == 0) {
                System.out.println("Training progress: " + (double) i / epochs * 100 + "%");
            }
        }

        // Test the neural network
        int correct = 0;
        double threshold = 0.5; // You can adjust the threshold for classification
        for (int i = 0; i < testSize; i++) {
            double[] output = nn.feedForward(testInputs.get(i));
            int predictedClass = output[0] >= threshold ? 1 : 0;
            int actualClass = (int) testTargets.get(i)[0];

            if (predictedClass == actualClass) {
                correct++;
            }
        }

        double accuracy = (double) correct / testSize * 100;
        System.out.println("Test accuracy: " + accuracy + "%");

    }
}