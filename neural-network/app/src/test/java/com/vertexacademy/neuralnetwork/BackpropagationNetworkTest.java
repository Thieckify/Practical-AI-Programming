package com.vertexacademy.neuralnetwork;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertSame;

public class BackpropagationNetworkTest {

    @Test
    public void outputLayerAndOutputsAreTracked() {
        BackpropagationNetwork network = new BackpropagationNetwork();
        NeuronLayer hiddenLayer = new NeuronLayer(2, 2);
        NeuronLayer outputLayer = new NeuronLayer(1, 2);

        network.addLayer(hiddenLayer);
        network.addLayer(outputLayer);

        int[] result = network.feed(new double[]{0.2, 0.8});

        assertEquals(1, result.length);
        assertSame(outputLayer, network.getOutputLayer());
        assertNotNull(network.getOutput());
        assertEquals(1, network.getOutput().length);
    }

    @Test
    public void trainingUpdatesWeightsAndKeepsOutputVector() {
        BackpropagationNetwork network = new BackpropagationNetwork();
        network.configure(1, 2, 2, 1);
        network.setLearningRate(0.5);

        network.train(new double[]{0.0, 1.0}, new double[]{1.0});

        assertEquals(1, network.getOutput().length);
        assertEquals(1, network.getOutputLayer().getNeurons().length);
    }
}
