#ifndef MM_COMMON_HPP
#define MM_COMMON_HPP
namespace mm{
struct GroundPoint {
    double z;
    double d;

    GroundPoint() : z(0), d(0) {}
    GroundPoint(const double& d, const double& z) : z(z), d(d) {}
    bool operator==(const GroundPoint& comp) {return z == comp.z && d == comp.d;}
};

struct GPModelParams{
	/* data */
	double l_scale;
	double sigma_f;
	double sigma_n;
    
	GPModelParams() : l_scale(28.01), sigma_f(1.76), sigma_n(0.12){}
};

struct GPPreParams {
	//GPPredictor thresholds
	double sigma_x;
	float  model_thr;
	float  data_thr;

	//constructor
	GPPreParams() : sigma_x(0.12), model_thr(0.2), data_thr(1.5){}
};

struct MMGroundSegmentationParams {
    // Visualize estimated ground.
    bool visualize;
    // Minimum range of segmentation.
    double r_min;
    // Maximum range of segmentation.
    double r_max;
    // Length of radial bins.
    double bin_length;
    // Number of angular segments.
    int n_segments;
    // Maximum distance to a ground line to be classified as ground.
    double max_dist_to_line;
    // Max slope to be considered ground line.
    double max_slope;
    // Max error for line fit.
    double max_error_square;
    // Distance at which points are considered far from each other.
    double long_threshold;
    // Maximum slope for
    double max_long_height;
    // Maximum heigh of starting line to be labelled ground.
    double max_start_height;
    // Height of sensor above ground.
    double sensor_height;
    // How far to search for a line in angular direction [rad].
    double line_search_angle;
    // Number of threads.
    int n_threads;

    MMGroundSegmentationParams(){}
};
}
#endif