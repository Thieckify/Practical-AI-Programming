package com.vertexacademy.neuralnetwork;

public class BackpropagationNetwork extends NeuronNetwork {

    private double deltaError;
    private double[] outputs;

    public BackpropagationNetwork() {
        super();
    }

    @Override
    public int[] feed(double[] inputs) {
        if (layers.isEmpty()) {
            throw new IllegalStateException("Le reseau de neurones ne contient aucune couche.");
        }

        double[] activations = evaluateActivations(inputs);
        outputs = activations;

        int[] binaryOutputs = new int[activations.length];
        for (int i = 0; i < activations.length; i++) {
            binaryOutputs[i] = activations[i] >= 0.5 ? 1 : 0;
        }

        deltaError = 0.0;
        for (int i = 0; i < activations.length; i++) {
            deltaError += Math.abs(binaryOutputs[i] - activations[i]);
        }

        return binaryOutputs;
    }

    // This is a recursive method. The backpropagation begins with the output layer
    public void train(double[] inputs, double[] desiredOutputs) {

        if (layers.isEmpty()) {
            throw new IllegalStateException("Le reseau de neurones ne contient aucune couche.");
        }

        if (desiredOutputs == null) {
            throw new IllegalArgumentException("Les sorties desirees ne doivent pas etre nulles.");
        }

        if (desiredOutputs.length != getNumberOfOutputs()) {
            throw new IllegalArgumentException("Le nombre de sorties desirees doit correspondre au nombre de neurones de sortie.");
        }

        /*
            Cette section prepare la propogation arriere, elle passe en avant toutes les couches pour
            memoriser :
                * Les activations de chaque couche
                * Les deltas (erreurs locales) pour chaque neurone.
            Confere cours etape deux de l'algorithme de retropropagation.
        */


        // Creer deux tableaux 2D.
        double[][] activations  = new double[layers.size()][];
        double[][] deltas       = new double[layers.size()][];

        // On stocke directement les entrees du reseau, ce qui correspond
        // a la couche d'entree.
        activations[0] = inputs.clone();

        // On parcourt toutes les couches du reseau, 
        // si c'est la premiere, l'entree provient directement de inputs
        // sinon l'entree de cette couche est la sortie de calculee par la
        // couche precedente.
        for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {

            NeuronLayer layer               = layers.get(layerIndex);
            double[] previousActivations    = (layerIndex == 0) ? inputs : activations[layerIndex - 1];

            SigmoidNeuron[] neurons = layer.getNeurons();

            // contient les sorties continue de la couche a layerIndex
            activations[layerIndex] = new double[neurons.length];

            // contiendra ensuite le gradient de cette couche lors de la retropropagation
            deltas[layerIndex]      = new double[neurons.length];

            // Pour chaque neuron on le feed avec les activation de la couche precedente
            // on recupere sa sortie continue et on la stocke dans activations[lauerIndex][neuronIndex]
            for (int neuronIndex = 0; neuronIndex < neurons.length; neuronIndex++) {
                neurons[neuronIndex].feed(previousActivations);
                activations[layerIndex][neuronIndex] = neurons[neuronIndex].getContinuousOutput();
            }
        }

        double learningRate             = layers.getFirst().getLearningRate();
        NeuronLayer outputLayer         = getOutputLayer();
        SigmoidNeuron[] outputNeurons   = outputLayer.getNeurons();

        for (int i = 0; i < outputNeurons.length; i++) {

            double activation = activations[activations.length - 1][i];
            deltas[deltas.length - 1][i] = (desiredOutputs[i] - activation) * activation * (1.0 - activation);

            double[] weights = outputNeurons[i].getWeights();
            double[] previousActivations = activations[activations.length - 2];
            for (int j = 0; j < weights.length; j++) {
                weights[j] += learningRate * deltas[deltas.length - 1][i] * previousActivations[j];
            }

            outputNeurons[i].setBias(outputNeurons[i].getBias() + learningRate * deltas[deltas.length - 1][i]);
        }

        for (int layerIndex = layers.size() - 2; layerIndex >= 0; layerIndex--) {
            NeuronLayer currentLayer = layers.get(layerIndex);
            SigmoidNeuron[] currentNeurons = currentLayer.getNeurons();
            SigmoidNeuron[] nextNeurons = layers.get(layerIndex + 1).getNeurons();

            double[] previousActivations = (layerIndex == 0) ? inputs : activations[layerIndex - 1];

            // Update the weights of the neurons using the initial inputs
            for (int neuronIndex = 0; neuronIndex < currentNeurons.length; neuronIndex++) {
                double sum = 0.0;
                for (int nextIndex = 0; nextIndex < nextNeurons.length; nextIndex++) {
                    sum += nextNeurons[nextIndex].getWeights()[neuronIndex] * deltas[layerIndex + 1][nextIndex];
                }

                double activation = activations[layerIndex][neuronIndex];
                deltas[layerIndex][neuronIndex] = sum * activation * (1.0 - activation);

                double[] weights = currentNeurons[neuronIndex].getWeights();
                for (int j = 0; j < weights.length; j++) {
                    weights[j] += learningRate * deltas[layerIndex][neuronIndex] * previousActivations[j];
                }

                currentNeurons[neuronIndex].setBias(currentNeurons[neuronIndex].getBias() + learningRate * deltas[layerIndex][neuronIndex]);
            }
        }

        feed(inputs);
    }

    private double[] evaluateActivations(double[] inputs) {
        double[] previous = inputs.clone();

        for (NeuronLayer layer : layers) {
            SigmoidNeuron[] neurons = layer.getNeurons();
            double[] next = new double[neurons.length];

            for (int i = 0; i < neurons.length; i++) {
                neurons[i].feed(previous);
                next[i] = neurons[i].getContinuousOutput();
            }

            previous = next;
        }

        return previous;
    }

    // Return the delta value computed when propagating
    // the error
    public double getDelta() {
        return deltaError;
    }

    // Return the output value, previously computed when doing feed
    public double[] getOutput() {
        return outputs;
    }

    // Return the output layer, which is also the last layer
    public NeuronLayer getOutputLayer() {
        for (NeuronLayer layer : this.layers) {
            if (layer.isOutputLayer()) {
                return layer;
            }
        }
        return null;
    }
}
