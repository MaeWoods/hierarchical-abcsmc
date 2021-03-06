/*
 * HeunSDESolver.hpp
 * Class defining an SDE Solver using the Heun approximation
 *
 *  Created on: Mar 14, 2011
 *      Author: student1
 */

#ifndef HEUNSDESOLVER_HPP_
#define HEUNSDESOLVER_HPP_

#include "SDESolver.hpp"

class HeunSDESolver: public SDESolver {
public:
	/**
	 * Array of size NSPECIES containing a intermediate value of the concentrations of each specie used for the algorithm
	 */
	double* intermediate_y;

	/**
	 * Array of size NSPECIES containing a intermediate value of the derivative of the concentrations of each specie used for the algorithm
	 */
	double* intermediate_dy;

	/**
	 * Array of size NSPECIES containing a intermediate value of the ODE part of the derivative of the concentrations of each specie used for the algorithm
	 */
	double* intermediate_dyODE;

	/**
	 * Array of size NSPECIES containing a intermediate value of the Euler part of the derivative of the concentrations of each specie used for the algorithm
	 */
	double* intermediate_dyEulerNoise;

	/**
	 * Constructor - Initialise the attributes specific to this algorithm - Runs the simulations
	 *
	 * @param double* ainitialValues array of size NSPECIES containing the initial values wanted to solve the system
	 * @param aparameters double* array of the parameters values wanted to solve the system
	 * @param Model* mmodel model we want to solve
	 * @param int inbOfIterations number of time we want to simulate the system
	 * @param vector<double> vtimepoints vector of the timepoints when we want the data to be computed and stored
	 * @param string sfilename name of the file we want to print the output
	 * @param double ddt time step wanted
	 * @return void
	 */
	HeunSDESolver(double* ainitialValues, double* aparameters, ChildModel* mmodel,
			int inbOfIterations, vector<double> vtimepoints, string sfilename,
			double ddt);

	/**
	 * Destructor
	 */
	~HeunSDESolver();

	/**
	 * Method returning a vector of size NREACTIONS giving the noise for each reaction that will be used for this step
	 *
	 * @param void
	 * @return vector<double> noise vector of size NREACTIONS giving the noise for each reaction that will be used for this step
	 */
	vector<double> getNoise();

	/**
	 * Method returning the Euler approximation part of the derivative of the concentration
	 *
	 * @param ColumnVector hazards Array of size NREACTIONS containing the hazards of all the reactions of the system
	 * @param vector<double>& noise vector of size NREACTIONS containing the noises associated at each reaction
	 * @return Array of size NSPECIES containing the Euler approximation part of the derivative of the concentrations
	 */
	double* getEulerNoiseTerm(const ColumnVector& hazards,
			const vector<double>& noise);

	/**
	 * Method checking at the end of each step the validity of the output (sometimes the steps has to be done again to have a good value because of the noise part)
	 *
	 * @param double* candidateDy the derivative we want to check the validity of
	 * @return 0 if the step has to be done again and 1 if the candidateDy is fine and can be used to increment the concentrations
	 */
	int checkY(double* candidateDy);

	/**
	 * Increment the concentrations at the end of one step
	 *
	 * @param void
	 * @return void
	 */
	void changeConc();

	/**
	 * Method doing one simulation of the system
	 *
	 * @param void
	 * @return void
	 */
	void step();
};

#endif /* HEUNSDESOLVER_HPP_ */
