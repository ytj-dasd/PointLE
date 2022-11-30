#ifndef COMMON_ZS
#define COMMON_ZS

#include <pcl/point_types.h>

struct PointXYZLO {
    float x = 0.f;
    float y = 0.f;
    float z = 0.f;
    int label = 0;
    int object = 0;

    PointXYZLO(){}
    PointXYZLO(float x, float y, float z, int label, int object) :
        x(x), y(y), z(z), label(label), object(object) {}
EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;


POINT_CLOUD_REGISTER_POINT_STRUCT(
    PointXYZLO,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (int, label, label)
    (int, object, object)
    )


#endif