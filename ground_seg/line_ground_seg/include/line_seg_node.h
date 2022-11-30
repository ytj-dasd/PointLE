#ifndef lINE_SEGNODE
#define LINE_SEGNODE

#include <ros/ros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>

#include "common_zs.h"
#include "ground_segmentation/ground_segmentation.h"

namespace line{
class  LineSegNode{

public:
    LineSegNode(ros::NodeHandle& node);
    ~LineSegNode();
    void groundSeg(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_in,
                   pcl::PointCloud<pcl::PointXYZ>& cloud_obstacle,
                   pcl::PointCloud<pcl::PointXYZ>& cloud_ground);

    void groundSeg(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_in,
                   pcl::PointCloud<PointXYZLO>& cloud_label);

private:
    std::string getLidarTopicName(ros::NodeHandle& node);
    std::string getGroundTopicName(ros::NodeHandle& node);
    std::string getObstacleTopicName(ros::NodeHandle& node);
    void getGroundSegmentationParams(ros::NodeHandle& node);

    void pointCloudCallback(const sensor_msgs::PointCloud2& cloud_msg);
    void processCloud(const sensor_msgs::PointCloud2& cloud_msg,
                      pcl::PointCloud<pcl::PointXYZ>& cloud_out); 
private:
    /* data */
    ros::Subscriber _lidar_sub;
    ros::Publisher _ground_pub;
    ros::Publisher _obstacle_pub;

    GroundSegmentationParams _ground_segmentator_params;
    float _filter_heigh = 0.1;
    float _filter_heigh_low = -0.1;
    float _ground_value = 0.02;

    int _frame_num = 0;
};
}
#endif