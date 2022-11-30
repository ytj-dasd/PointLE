#ifndef MM_GROUND_SEGMENTATION_H_
#define MM_GROUND_SEGMENTATION_H_

#include <mutex>

#include <glog/logging.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "common.hpp"
#include "segment.h"

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef std::pair<pcl::PointXYZ, pcl::PointXYZ> PointLine;

namespace mm{
class MMGroundSegmentation {

public:
    MMGroundSegmentation(const MMGroundSegmentationParams& params = MMGroundSegmentationParams(),
                         const GPModelParams& gp_model_params = GPModelParams(),
                         const GPPreParams& gp_pre_params = GPPreParams());

    void segment(const PointCloud& cloud, std::vector<int>* segmentation);

private:
    void assignCluster(std::vector<int>* segmentation);

    void assignClusterThread(const unsigned int& start_index,
                              const unsigned int& end_index,
                              std::vector<int>* segmentation);

    void insertPoints(const PointCloud& cloud);

    void insertionThread(const PointCloud& cloud,
                         const size_t start_index,
                         const size_t end_index);

    void getGroundPoints(PointCloud* out_cloud);

    void getLines(std::list<PointLine>* lines);

    void lineFitThread(const unsigned int start_index, const unsigned int end_index,
                       std::list<PointLine> *lines, std::mutex* lines_mutex);

    pcl::PointXYZ GroundPointTo3d(const GroundPoint& min_z_point, const double& angle);

    void getGroundPointCloud(PointCloud* cloud);

    void visualizePointCloud(const PointCloud::ConstPtr& cloud,
                             const std::string& id = "point_cloud");

    void visualizeLines(const std::list<PointLine>& lines);

    void visualize(const std::list<PointLine>& lines, 
                   const PointCloud::ConstPtr& cloud, 
                   const PointCloud::ConstPtr& ground_cloud, 
                   const PointCloud::ConstPtr& obstacle_cloud);
    void saveLineGroundPoint(std::string file_path, const std::vector<int>* segmentation);
    void saveSegmentPoint(std::string file_path);

private:
    const MMGroundSegmentationParams _seg_params;
    const GPModelParams _gp_model_params;
    const GPPreParams _gp_pre_params;

    // Access with _segments[segment][bin].
    std::vector<Segment> _segments;

    // Bin index of every point.
    std::vector<std::pair<int, int> > _bin_index;

    // 2D coordinates (d, z) of every point in its respective segment.
    std::vector<GroundPoint> _segment_coordinates;

    // Visualizer.
    std::shared_ptr<pcl::visualization::PCLVisualizer> _viewer;
};
}
#endif // GROUND_SEGMENTATION_H_
