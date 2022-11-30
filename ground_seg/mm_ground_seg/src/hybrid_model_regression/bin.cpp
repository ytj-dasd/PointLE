#include "hybrid_model_regression/bin.h"

#include <limits>
namespace mm{
Bin::Bin() : _min_z(std::numeric_limits<double>::max()), _has_point(false) {}

Bin::Bin(const Bin& bin) {
    if (bin.hasPoint()) {
        _has_point.store(true);
        _min_z.store(bin.getGroundPoint().z);
        _min_z_range.store(bin.getGroundPoint().d);
        return;
    }
    _min_z.store(std::numeric_limits<double>::max());
    _min_z_range.store(0);
    _has_point.store(false);
}

// Bin::Bin(const Bin& bin) : _min_z(std::numeric_limits<double>::max()),
//                             _has_point(false) {}

void Bin::addPoint(const pcl::PointXYZ& point) {
    const double d = sqrt(point.x * point.x + point.y * point.y);
    addPoint(d, point.z);
}

void Bin::addPoint(const double& d, const double& z) {
    _has_point.store(true);
    if (z < _min_z) {
      _min_z .store(z);
      _min_z_range.store(d);
    }
}

void Bin::addGroundPoint(const GroundPoint& ground_point) {
    if (!_has_point.load()) {
        _min_z.store(ground_point.z);
        _min_z_range.store(ground_point.d);
        _has_point.store(true);
    }
}

GroundPoint Bin::getGroundPoint() const{
    GroundPoint point;
    if (_has_point.load()) {
        point.z = _min_z.load();
        point.d = _min_z_range.load();
    }
    return point;
}
}