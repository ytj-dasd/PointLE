#ifndef MM_GPPREDICTOR_H
#define MM_GPPREDICTOR_H

#include <Eigen/Core>

#include "common.hpp"
#include "gaussian_process.hpp"
#include "segment.h"

///************************************************************************///
// a class denotes the GP-GPPredictor algorithm (i.e.,GPPredictor part)
//
// the implementation refers to this paper as below:
// Douillard B, Underwood J, Kuntz N, et al. On the segmentation of 3D LIDAR point clouds,
// IEEE International Conference on Robotics and Automation. IEEE, 2011:2798-2805.
//
// Generated and edited by Zhangshen 2020.10.18
///************************************************************************///

namespace mm{
class GPPredictor {

public:
	typedef Eigen::Matrix<float, 1, 1> GroundD;
	typedef Eigen::Matrix<float, 1, 1> GroundZ;
	typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> MatrixXr;

	GPPredictor(GPModelParams gp_model_params, GPPreParams gp_pre_params);
	~GPPredictor();

	void predict(const std::vector<std::vector<Bin> >& ground_skeletons, 
		         const std::vector<GroundPoint>& segment_coordinates,
	             const std::vector<std::vector<int> >& segs_obstacle_pt_index,
			     std::vector<int>* segmentation, bool use_thread = false) const;
private:
	//to make a point clouds as the train input of GaussianProcessRegression
	void toAddTrainSamples(const std::vector<Bin>& segment_skeleton,
						   MatrixXr& train_x_vec, MatrixXr& train_y_vec) const;

	//to make a point clouds as the test input of GaussianProcessRegression
	void toAddTestSamples(const std::vector<int>& obstacle_pt_index,
		                  const std::vector<GroundPoint>& segment_coordinates,
						  std::vector<GroundD>& test_x_vec, \
						  std::vector<GroundZ>& test_y_vec) const;

    int eval(const float& fZValue, const float& fMean, const float& fVar) const;

    void predictThreadPool(const std::vector<std::vector<Bin> >& ground_skeletons, 
						   const std::vector<GroundPoint>& segment_coordinates,
						   const std::vector<std::vector<int> >& segs_obstacle_pt_index,
						   std::vector<int>* segmentation) const;
	
    void predictThread(const std::vector<std::vector<Bin> >& ground_skeletons, 
					   const std::vector<GroundPoint>& segment_coordinates,
					   const std::vector<std::vector<int> >& segs_obstacle_pt_index,
					   std::vector<int>* segmentation, int start, int end) const;
private:
	const GPModelParams _gp_model_params;
	const GPPreParams _gp_pre_params;
};
}
#endif // !GPPREDICTOR_H
