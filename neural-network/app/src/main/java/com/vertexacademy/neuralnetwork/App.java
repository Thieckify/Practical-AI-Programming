/*
 * @File    : App.java
 * @Author  : Yossep BINYOUM
 * @Date    : 07/2026
 * @Brief   : Classe de demonstration (point d'entree) permettant de :
 * 
 *              1) Tester un SigmoidNeuron seul : creation, feed(),
 *                 puis quelques iterations d'entrainement sur une
 *                 porte logique OR pour verifier que la sortie
 *                 continue se rapproche bien des valeurs attendues.
 *              2) Construire une NeuronLayer et verifier que chaque
 *                 neurone de la couche produit bien une sortie.
*/
package com.vertexacademy.neuralnetwork;

public class App {

    private static double[][] inputs = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    // Entrainement sur plusieurs epoques
    private static int nbEpochs = 1000;

    public static void main(String[] args) {

        //testNeuron();
        //testSigmoidNeuron();
        //testNeuronLayer();
        testDigitalComparatorLearning();
    }

    /*
     *  Teste un Perceptron seul : on initialise avec des poids et bias
     *  connus, on observe sa sortie avant entrainement. 
     *  Puis on l'entraine sur la table de verite de la porte OR.
    */

    private static void testNeuron() {

        System.out.println("==== Test de Perceptron =====\n");

        Neuron neuron = new Neuron(2);

        neuron.setWeights(new double[]{0.2, 0.2});
        neuron.setBias(-0.3);

        double[] desiredOutputs = {0, 1, 1, 1}; // porte OR
        
        System.out.println("Avant entrainement :");
        System.out.println();
        
        afficherParametres(neuron, 0);
        afficherSorties(neuron, inputs);


        for (int epoch = 0; epoch < nbEpochs; epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                neuron.train(inputs[i], desiredOutputs[i]);   
            }

            if ((epoch + 1) % 100 == 0) {
                afficherParametres(neuron, epoch + 1);
            }
        }
    
        System.out.println("\nAprès " + nbEpochs + " epoques d'entrainement :");
        System.out.println();

        afficherSorties(neuron, inputs);
        System.out.println();
    }

    /*
    *   Teste un SigmoidNeuron seul : on l'initialise avec des poids
    *   et un biais connus, on observe sa sortie avant entrainement,
    *   puis on l'entraine sur la table de verite de la porte OR
    *   afin de verifier que sa sortie continue converge bien vers
    *   les valeurs desirees (proche de 0 ou proche de 1).
    */
    private static void testSigmoidNeuron() {

        System.out.println("=== Test de SigmoidNeuron ===\n");

        SigmoidNeuron neuron = new SigmoidNeuron(2);
        neuron.setWeights(new double[]{0.5, 0.5});
        neuron.setBias(0.0);
        neuron.setLearningRate(0.5);

        
        double[] desiredOutputs = {0, 1, 1, 1}; // porte OR

        System.out.println("Avant entrainement :");
        System.out.println();
        afficherSorties(neuron, inputs);

        
        for (int epoch = 0; epoch < nbEpochs; epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                neuron.train(inputs[i], desiredOutputs[i]);
            }
        }

        System.out.println("\nAprès " + nbEpochs + " epoques d'entrainement :");
        System.out.println();

        afficherSorties(neuron , inputs);
        System.out.println();
    }

    /*
    *   Construit une NeuronLayer et verifie que chaque neurone de la
    *   couche produit bien une sortie lorsqu'on lui transmet des
    *   entrees. Comme les poids/biais sont initialises aleatoirement,
    *   ce test verifie surtout la bonne construction de la couche et
    *   la propagation correcte a travers tous ses neurones.
    */
    private static void testNeuronLayer() {

        System.out.println("=== Test de NeuronLayer ===\n");

        int nbOfNeurons = 4;
        int nbOfWeights = 2;


        NeuronLayer layer = new NeuronLayer(nbOfNeurons, nbOfWeights);
        layer.setNextLayer(null);

        System.out.println("Couche construite avec " + layer.getNeurons().length
                        + " neurones, chacun avec " + nbOfWeights + " poids.");
        System.out.println("learningRate de la couche = " + layer.getLearningRate());
        
        
        // Entrainement de la couche de neurones sur la table de verite de la porte OR
        double[] desiredOutputs = {0, 1, 1, 1}; // porte OR

        System.out.println("\nEtat initial :");
        afficherEtatCouche(layer, inputs, desiredOutputs, 0);

        for (int epoch = 0; epoch < nbEpochs; epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                layer.train(inputs[i], desiredOutputs);
            }

            if ((epoch + 1) % 100 == 0) {
                afficherEtatCouche(layer, inputs, desiredOutputs, epoch + 1);
            }
        }

        System.out.println("\nEtat final :");
        afficherEtatCouche(layer, inputs, desiredOutputs, nbEpochs);
    }

    private static void testDigitalComparatorLearning() {
        System.out.println("=== Apprentissage du comparateur digital ===\n");

        BackpropagationNetwork network = new BackpropagationNetwork();
        network.configure(1, 8, 2, 3);
        network.setLearningRate(0.5);

        double[][] inputs = {
            {0.0, 0.0},
            {0.0, 1.0},
            {1.0, 0.0},
            {1.0, 1.0}
        };

        double[][] desired = {
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0},
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0}
        };

        int epochs = 6000;
        for (int epoch = 1; epoch <= epochs; epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                network.train(inputs[i], desired[i]);
            }

            if (epoch == 1 || epoch == 10 || epoch == 100 || epoch == 1000 || epoch == epochs) {
                System.out.printf("Epoch %d | erreur moyenne = %.6f%n", epoch, computeMeanError(network, inputs, desired));
            }
        }

        System.out.println();
        System.out.println("Sorties finales apres apprentissage :");
        for (int i = 0; i < inputs.length; i++) {
            int[] result = network.feed(inputs[i]);
            System.out.printf("A=%.0f, B=%.0f -> [A>B, A=B, A<B] = [%d, %d, %d]%n",
                    inputs[i][0], inputs[i][1], result[0], result[1], result[2]);
        }

        System.out.println();
        System.out.println("Le reseau apprend les 4 cas du comparateur numerique avec des couches cachees entrainables.");
    }

    private static double computeMeanError(BackpropagationNetwork network, double[][] inputs, double[][] desiredOutputs) {
        double total = 0.0;

        for (int i = 0; i < inputs.length; i++) {
            int[] predicted = network.feed(inputs[i]);
            for (int j = 0; j < desiredOutputs[i].length; j++) {
                total += Math.abs(desiredOutputs[i][j] - predicted[j]);
            }
        }

        return total / (inputs.length * desiredOutputs[0].length);
    }


    /*
    *   Affiche, pour chaque entree, la sortie continue (sigmoide)
    *   ainsi que la sortie binaire (seuillee a 0.5) du neurone.
    */
    private static void afficherSorties(SigmoidNeuron neuron, double[][] inputs) {
        
        for (double[] input : inputs) {

            int binaryOutput = neuron.feed(input);
            double continuousOutput = neuron.getContinuousOutput();
            System.out.printf("  [%.0f, %.0f] -> sortie continue = %.4f | sortie binaire = %d%n",
                    input[0], input[1], continuousOutput, binaryOutput);
        }
    }

    private static void afficherSorties(Neuron neuron, double[][] inputs) {

        for (double[] input : inputs) {
            int binaryOutput = neuron.feed(input);
            System.out.printf("  [%.0f, %.0f] -> sortie binaire = %d%n",
                    input[0], input[1], binaryOutput);
        }
    }

    private static void afficherParametres(Neuron neuron, int epoch) {
        
        double[] weights = neuron.getWeights();
        System.out.printf("  Epoque %d : poids = [", epoch);

        for (int i = 0; i < weights.length; i++) {
            System.out.printf("%.4f%s", weights[i], i < weights.length - 1 ? ", " : "");
        }
        System.out.printf("] | bias = %.4f%n", neuron.getBias());
    }

    private static void afficherEtatCouche(
        NeuronLayer layer, 
        double[][] inputs,
        double[] desiredOutputs, 
        int epoch
    ) {
        System.out.printf("\n--- Epoque %d ---%n", epoch);

        SigmoidNeuron[] neurons = layer.getNeurons();

        for (int neuronIndex = 0; neuronIndex < neurons.length; neuronIndex++) {

            SigmoidNeuron neuron = neurons[neuronIndex];
            double[] weights = neuron.getWeights();

            System.out.printf("Neurone %d : poids = [", neuronIndex);
            for (int weightIndex = 0; weightIndex < weights.length; weightIndex++) {
                System.out.printf("%.4f%s", weights[weightIndex],
                                  weightIndex < weights.length - 1 ? ", " : "");
            }
            System.out.printf("] | biais = %.4f%n", neuron.getBias());

            for (int inputIndex = 0; inputIndex < inputs.length; inputIndex++) {

                int binaryOutput = neuron.feed(inputs[inputIndex]);
                double continuousOutput = neuron.getContinuousOutput();
                double error = desiredOutputs[neuronIndex] - continuousOutput;

                System.out.printf("  entree [%d, %d] -> cible = %.0f\t| Pourcentage activation = %.4f\t| "
                                  + "binaire = %d\t| erreur = %+.4f%n",
                                  (int) inputs[inputIndex][0], (int) inputs[inputIndex][1],
                                  desiredOutputs[neuronIndex], continuousOutput * 100.0f,
                                  binaryOutput, error);
            }
            System.out.println();
        }
    }
}