package com.vertexacademy.neuralnetwork;


import java.util.ArrayList;
import java.util.List;

public class NeuronNetwork {

    //
    //  Instances variables
    //

    // Layers un tableau dynamique de couche de neurones chainees.
    protected  List<NeuronLayer> layers = new ArrayList<>();

    // variable d'instance errors et precisions seront utiles pour suivre 
    // l'évolution des erreurs au cours de la phase d'apprentissage
    protected  double[] errors;
    protected  double[] precisions;


    //
    //  Public API
    //
    public NeuronNetwork(){}

    public void addLayer(NeuronLayer newLayer) {

        if (newLayer == null) {
            throw new IllegalArgumentException("newLayer ne doit pas être nul");
        }

        if (newLayer.getPreviousLayer() != null || newLayer.getNextLayer() != null) {
            throw new IllegalArgumentException("newLayer ne doit pas deja etre connectee");
        }

        // Si le reseau est vide ajoute tout simplement la couche sans creer de 
        // connexion
        if (layers.isEmpty()) {
            layers.add(newLayer);
            return;
        }

        // On cree la connexion entre les differentes couches
        NeuronLayer previousLayer = layers.getLast();
        newLayer.setPreviousLayer(previousLayer);
        previousLayer.setNextLayer(newLayer);
        layers.add(newLayer);
    }

    // Feeding a neural network involves simply feeding the first hidden layer
    public int[] feed(double[] inputs) {

        NeuronLayer firstLayer = this.layers.getFirst();

        if(firstLayer == null) {
            throw new IllegalStateException("Impossible de feed une couche nulle");
        }
        
        return firstLayer.feed(inputs);
    }

    // Retourne le nombre de sortie du reseau neuronal
    public int getNumberOfOutputs() {

        if (layers.isEmpty()) {
            throw new IllegalStateException("Le reseau de neurones ne contient aucune couche.");
        }

        NeuronLayer lastLayer = layers.getLast();

        if (lastLayer.isOutputLayer()) {
            return lastLayer.getNeurons().length;
        }
        else {
            return 0;
        }
    }

    /*
        Deux methodes qui permettent de configurer le reseau neuronal avec des
        parametres bien definis. Ceci est interessant dans le cas ou on veut 
        creer un reseau neuronal avec une couche cachee et une couche de sortie.
        Ou tout simplement x nombre de couches cachees.
    */

    // Cree un reseau avec une couche cache contenant 2 neurons, nombre de poids 1 par neurones
    // et une couche de sortie.
    public void configure (int nbOfHiddenLayers, int nbOfNeurons, int nbOfWeights, int nbOfOutput) {
        
        if (!layers.isEmpty()) {
            throw new IllegalStateException("Le reseau de neurones est deja configure.");
        }

        if (nbOfHiddenLayers < 0 || nbOfNeurons <= 0 || nbOfWeights <= 0 || nbOfOutput <= 0) {
            throw new IllegalArgumentException("Les parametres de configuration sont invalides.");
        }

        for (int i = 0; i < nbOfHiddenLayers; i++) {
            int numberOfWeights = i == 0 ? nbOfWeights : nbOfNeurons;
            addLayer(new NeuronLayer(nbOfNeurons, numberOfWeights));
        }

        // Couche de sortie
        NeuronLayer outputLayer = new NeuronLayer(nbOfOutput, nbOfNeurons);
        addLayer(outputLayer);
    }

    //
    // Setters
    //

    // Definit le taux d'apprentissage pour toutes les couches du reseau neuronal
    public void setLearningRate(double learningRate) {

        if (layers.isEmpty()) 
            throw new IllegalStateException("Impossible d'ajuster le taux d'apprentissage" + 
            "d'un reseau neuronal vide.");
        
        for (NeuronLayer layer : layers) {
            layer.setLearningRate(learningRate);
        }
    }

    public static void main(String[] args) {
        NeuronNetwork digitalComparatorNetwork = new NeuronNetwork();
        digitalComparatorNetwork.configure(1, 4, 2, 3);

        // Comme vous pouvez le constatez, notre reseau neuronal ne se comporte pas encore 
        // comme un comparateur digital 
        //  A  B       A<B      A = B    A>B
        // [0, 0] => [aLb = 0, aEb = 1, aGb = 0]
        // [0, 1] => [alb = 1, aEb = 0, aGb = 0]
        // [1, 0] => [aLb = 0, aEb = 0, aGb = 1]
        // [1, 1] => [aLb = 0, aEb = 1, aGb = 0]

        int[] outputs = digitalComparatorNetwork.feed(new double[]{0, 1});
        System.out.printf("A = 0 and B = 1\n[aLb] = %d, [aEb] = %d, [aGb] = %d\n", 
                        outputs[0], outputs[1], outputs[2]);
    }
}