
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <ctime>
#include "ground_extraction/GaussianProcess.h"

int main() {

	GaussianProcessRegression<float> gpr(1, 1);
	gpr.SetHyperParams(1.16, 1.268, 0.3);
	typedef Eigen::Matrix<float, 1, 1> input_type;
	typedef Eigen::Matrix<float, 1, 1> output_type;
	std::vector<input_type> test_inputs;//train_inputs, 
	std::vector<output_type> test_outputs;//train_outputs, 

    auto gp1_start_t = std::chrono::steady_clock::now();
	Eigen::Matrix<float, 1, 4> train_inputs;
	train_inputs(0) = -1.50;
	train_inputs(1) = -1.00;
	train_inputs(2) = -0.75;
	train_inputs(3) = -0.40;
	//train_inputs(4)=-0.25;
	//train_inputs(5)=0.00;
	//std::cout<<train_inputs<<std::endl<<"another:"<<std::endl;
	Eigen::Matrix<float, 1, 4> train_outputs;
	train_outputs(0) = -1.6;
	train_outputs(1) = -1.1;
	train_outputs(2) = -0.4;
	train_outputs(3) = 0.2;
	//train_outputs(4)=0.5;
	//train_outputs(5)=0.8;
	//std::cout<<train_outputs<<std::endl;
	gpr.AddTrainingDatas(train_inputs, train_outputs);

	Eigen::Matrix<float, 1, 1> oneTestInputs;
	oneTestInputs(0) = 0.2;
	test_inputs.push_back(oneTestInputs);

	Eigen::Matrix<float, Eigen::Dynamic, 1> vPredValue;
	Eigen::Matrix<float, Eigen::Dynamic, 1> vPredVar;
	for (size_t k = 0; k<test_inputs.size(); k++) {
		gpr.Regression(vPredValue, vPredVar, test_inputs[k]);
		// std::cout<<"y:"<<std::endl<<vPredValue<<std::endl;
		// std::cout << "var(y):" << std::endl << vPredVar << std::endl;
	}

	auto output_t = std::chrono::steady_clock::now();
	std::chrono::duration<double, std::micro> out_dur_num = output_t - gp1_start_t;
	std::cout << out_dur_num.count() << " us" << std::endl;


	std::vector<input_type> test_inputs2;//train_inputs, 
	std::vector<output_type> test_outputs2;//train_outputs, 

	Eigen::Matrix<float, 1, 2> train_inputs2;

	train_inputs2(0) = -0.25;
	train_inputs2(1) = 0.00;
	//std::cout << train_inputs2 << std::endl << "another:" << std::endl;
	Eigen::Matrix<float, 1, 2> train_outputs2;

	train_outputs2(0) = 0.5;
	train_outputs2(1) = 0.8;
	//std::cout << train_outputs2 << std::endl;
	gpr.AddTrainingDatas(train_inputs2, train_outputs2);

	Eigen::Matrix<float, 1, 1> oneTestInputs2;
	oneTestInputs2(0) = 0.2;
	test_inputs2.push_back(oneTestInputs2);
	oneTestInputs2(0) = 0.2;
	test_inputs2.push_back(oneTestInputs2);
	oneTestInputs2(0) = 0.4;
	test_inputs2.push_back(oneTestInputs2);
	oneTestInputs2(0) = 0.2;
	test_inputs2.push_back(oneTestInputs2);

	Eigen::Matrix<float, Eigen::Dynamic, 1> vPredValue2;
	Eigen::Matrix<float, Eigen::Dynamic, 1> vPredVar2;
	for (size_t k = 0; k<test_inputs2.size(); k++) {
		gpr.Regression(vPredValue2, vPredVar2, test_inputs2[k]);
		// std::cout << "y:" << std::endl << vPredValue2 << std::endl;
		// std::cout << "var(y):" << std::endl << vPredVar2 << std::endl;
	}

	return 0;

}
