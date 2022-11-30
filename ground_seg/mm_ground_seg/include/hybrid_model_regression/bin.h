#ifndef MM_GROUND_SEGMENTATION_BIN_H_
#define MM_GROUND_SEGMENTATION_BIN_H_

#include <atomic>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "common.hpp"
namespace mm{
class Bin {

public:
    Bin();
    /// \brief Fake copy constructor to allow vector<vector<Bin> > initialization.
    Bin(const Bin& bin);

    void addPoint(const pcl::PointXYZ& point);

    void addPoint(const double& d, const double& z);

    void addGroundPoint(const GroundPoint& ground_point);

    GroundPoint getGroundPoint() const;

    inline bool hasPoint() const {return _has_point.load();}

private:
    std::atomic<bool> _has_point;
    std::atomic<double> _min_z;
    std::atomic<double> _min_z_range;
};
}
#endif /* GROUND_SEGMENTATION_BIN_H_ */
