//
// Created by Im YoungMin on 10/28/19.
//

#ifndef MACHINE_LEARNING_NEURALNETWORK_H
#define MACHINE_LEARNING_NEURALNETWORK_H

#include <string>
#include "petsc_compatibility.h"

/**
 * Defines a neural model whose weights have been trained with Tensorflow in Python.
 * See https://github.com/imyoungmin/LSCurvatureML for more details.
 */
class NeuralNetwork
{
private:
	const std::string _ROOT_DIR = "/Users/youngmin/Documents/CS/CASL/LSCurvatureML/";
	const std::string _WEIGHTS_AND_BIASES_DIR = _ROOT_DIR + "models/WeightsAndBiases";
	const std::string _DATA_DIR = _ROOT_DIR + "data/";

public:
	NeuralNetwork( MPI_Comm mpicomm );
};


#endif //MACHINE_LEARNING_NEURALNETWORK_H
