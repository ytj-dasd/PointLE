#include <omp.h>
#include <thread>
#include "hybrid_model_regression/gp_predictor.h"

namespace mm{
GPPredictor::GPPredictor(GPModelParams gp_model_params, 
                         GPPreParams gp_pre_params) : 
						_gp_model_params(gp_model_params),
						_gp_pre_params(gp_pre_params){}

GPPredictor::~GPPredictor() {}

void GPPredictor::predict(const std::vector<std::vector<Bin> >& ground_skeletons, 
						  const std::vector<GroundPoint>& segment_coordinates,
						  const std::vector<std::vector<int> >& segs_obstacle_pt_index,
						  std::vector<int>* segmentation, bool use_thread) const {
    if (use_thread) {
	    predictThreadPool(ground_skeletons, segment_coordinates, 
	        segs_obstacle_pt_index, segmentation);
	    return;
	}
	omp_set_num_threads(3);
    #pragma omp parallel for
	for (int i = 0; i < ground_skeletons.size(); ++i) {
		GaussianProcessRegression<float> gpr(1, 1);
	    gpr.SetHyperParams(_gp_model_params.l_scale, _gp_model_params.sigma_f, 
		                   _gp_model_params.sigma_n);
		
		Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> train_x_vec;
		Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> train_y_vec;
		toAddTrainSamples(ground_skeletons[i], train_x_vec, train_y_vec);

        if (train_x_vec.size() == 0) {
			continue;
		}

		//new test vector - NEW output
		std::vector<Eigen::Matrix<float, 1, 1> > test_x_vec;
		std::vector<Eigen::Matrix<float, 1, 1> > test_y_vec;
		toAddTestSamples(segs_obstacle_pt_index[i], segment_coordinates,
			             test_x_vec, test_y_vec);
		
		gpr.AddTrainingDatas(train_x_vec, train_y_vec);
		//training
		bool keep_train = true;
		while(keep_train) {
			keep_train = false;
			for (size_t k = 0; k < test_x_vec.size(); k++) {
				if(segmentation->at(segs_obstacle_pt_index[i][k]) != 0){
					continue;
				}
				Eigen::Matrix<float, Eigen::Dynamic, 1> pred_z;
				Eigen::Matrix<float, Eigen::Dynamic, 1> pred_var;

				//regression of new input based on the computed training matrix
				if (gpr.Regression(pred_z, pred_var, test_x_vec[k])){
					int label = eval(test_y_vec[k](0), pred_z(0), pred_var(0));
					segmentation->at(segs_obstacle_pt_index[i][k]) = label;
					if (label == 1) {
						gpr.AddTrainingData(test_x_vec[k], test_y_vec[k]);
						keep_train = true;
					}
				}else{
					segmentation->at(segs_obstacle_pt_index[i][k]) = -1;  
				}
			}
		}
	}
}

void GPPredictor::toAddTrainSamples(const std::vector<Bin>& segment_skeleton,
					                MatrixXr& train_x_vec, MatrixXr& train_y_vec) const{
	int train_x_num = 0;
	for (size_t i = 0; i < segment_skeleton.size(); ++i) {
		if (!segment_skeleton[i].hasPoint()) {
			continue;
		}
		train_x_num++;
		train_x_vec.conservativeResize(1, train_x_num);
		train_y_vec.conservativeResize(1, train_x_num);
		train_x_vec(train_x_num - 1) = segment_skeleton[i].getGroundPoint().d;
		train_y_vec(train_x_num - 1) = segment_skeleton[i].getGroundPoint().z;
	}
}

void GPPredictor::toAddTestSamples(const std::vector<int>& obstacle_pt_index,
								   const std::vector<GroundPoint>& segment_coordinates,
								   std::vector<GroundD>& test_x_vec, 
								   std::vector<GroundZ>& test_y_vec) const{
    test_x_vec.clear();
    test_y_vec.clear();
    Eigen::Matrix<float, 1, 1> x;
    Eigen::Matrix<float, 1, 1> y;
    for (size_t i = 0; i < obstacle_pt_index.size(); ++i) {
		x(0) = segment_coordinates[obstacle_pt_index[i]].d;
		y(0) = segment_coordinates[obstacle_pt_index[i]].z;
	    test_x_vec.push_back(x);
	    test_y_vec.push_back(y);
	}
}

int GPPredictor::eval(const float & fZValue, const float & fMean, const float & fVar) const{

	//-1 indicates obstacle,0 indicates unknown point,1 indicates ground
	//first condition is to remove distant point
	if (std::fabs(fVar) < _gp_pre_params.model_thr) {
	    
		//Mahalanobis distance
		// float fGroundValue = fabs(fMean - fZValue) / 
		//     sqrt(_gp_pre_params.sigma_x * _gp_pre_params.sigma_x + fVar * fVar);

		// Euclidean distance 
		float fGroundValue = fabs(fMean - fZValue);
		//the second condition is to get ground point based on prediction value
		//it means that the GP algorithm predicts an ideal ground value
		//then a comparison is carried out between the ideal value and query value
		if (fGroundValue < _gp_pre_params.data_thr){ 
		    // std::cout << "fMean: " << fMean << ", fZValue: " << fZValue;
		    // std::cout << ", fVar: " << fVar << ", fGroundValue: "<< fGroundValue << std::endl;
			return 1;
		}else{
			return -1;
		}
	}else{
		//too far away so that it will be computed again later
		return 0;         
	}
}


/////////////////////////////////////////////////////////////////////////////
//multi thread
void GPPredictor::predictThreadPool(const std::vector<std::vector<Bin> >& ground_skeletons, 
						            const std::vector<GroundPoint>& segment_coordinates,
						            const std::vector<std::vector<int> >& segs_obstacle_pt_index,
						            std::vector<int>* segmentation) const {
   
	int thread_num = 3;
	int step = ground_skeletons.size() / thread_num;
	std::vector<std::thread> thread_vec(thread_num);
	int start = 0;
	int end = 0;
	for (int i = 0; i < thread_num - 1; ++i) {
		start = i * step;
		end = (i + 1) * step;
		thread_vec[i] = std::thread(&GPPredictor::predictThread, this, ground_skeletons,
		    segment_coordinates, segs_obstacle_pt_index, segmentation, start, end);
	}
    start = end;
	end = ground_skeletons.size();
	thread_vec[thread_num - 1] = std::thread(&GPPredictor::predictThread, this, ground_skeletons,
	    segment_coordinates, segs_obstacle_pt_index, segmentation, start, end);


    for (int i = 0; i < thread_num; ++i) {
		if (thread_vec[i].joinable()) {
			thread_vec[i].join();
		}
	}
}

void GPPredictor::predictThread(const std::vector<std::vector<Bin> >& ground_skeletons, 
						        const std::vector<GroundPoint>& segment_coordinates,
						        const std::vector<std::vector<int> >& segs_obstacle_pt_index,
						        std::vector<int>* segmentation, int start, int end) const {
	
	for (int i = start; i < end; ++i) {
		GaussianProcessRegression<float> gpr(1, 1);
	    gpr.SetHyperParams(_gp_model_params.l_scale, _gp_model_params.sigma_f, 
		                   _gp_model_params.sigma_n);
		
		Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> train_x_vec;
		Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> train_y_vec;
		toAddTrainSamples(ground_skeletons[i], train_x_vec, train_y_vec);

        if (train_x_vec.size() == 0) {
			continue;
		}
		//new test vector - NEW output
		std::vector<Eigen::Matrix<float, 1, 1> > test_x_vec;
		std::vector<Eigen::Matrix<float, 1, 1> > test_y_vec;
		toAddTestSamples(segs_obstacle_pt_index[i], segment_coordinates,
			             test_x_vec, test_y_vec);
		//training
		gpr.AddTrainingDatas(train_x_vec, train_y_vec);

		//regression (prediction)
		for (size_t k = 0; k < test_x_vec.size(); k++) {

			Eigen::Matrix<float, Eigen::Dynamic, 1> pred_z;
			Eigen::Matrix<float, Eigen::Dynamic, 1> pred_var;

			//regression of new input based on the computed training matrix
			if (gpr.Regression(pred_z, pred_var, test_x_vec[k])){
				int label = eval(test_y_vec[k](0), pred_z(0), pred_var(0));
				segmentation->at(segs_obstacle_pt_index[i][k]) = label;
			}else{
				segmentation->at(segs_obstacle_pt_index[i][k]) = -1;  //obstacle
			}
		}
	}
}
}