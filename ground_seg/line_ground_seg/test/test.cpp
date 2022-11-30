#include <ros/ros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <chrono>

#include "line_seg_node.h"

int main(int argc, char** argv) {
    
    ros::init(argc, argv, "line_gound_seg_test");
    ros::NodeHandle node("~");
    
    line::LineSegNode seg_node(node);
    ROS_INFO("------TEST-----------");
    std::string cloud_path = "/home/zs/zs/master/experiment/ground_seg/special_scene/jiayi/2430.ply";
    // std::string cloud_path = "/media/zs/TOSHIBA EXT/rosdata/cloud/ground/left_cloud_ply/20.ply";
    // std::string cloud_path = "/media/zs/TOSHIBA EXT/JRDB/jrdb_train/cvgl/group/jrdb/data/train_dataset/pointclouds/lower_velodyne/svl-meeting-gates-2-2019-04-08_1/000023.pcd";
    std::string cloud_save = "/home/zs/zs/kitti/data/2430_";
    ROS_INFO("cloud_path: %s", cloud_path.c_str());
    ROS_INFO("cloud_path: %s", cloud_save.c_str());
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<PointXYZLO>::Ptr cloud_label(new pcl::PointCloud<PointXYZLO>);
    pcl::io::loadPLYFile(cloud_path, *cloud);
    // pcl::io::loadPCDFile(cloud_path, *cloud);
    auto start = std::chrono::steady_clock::now();
    seg_node.groundSeg(cloud, *cloud_label);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> dur = end - start;
    ROS_INFO("run time: %f ms", dur.count());
    pcl::io::savePCDFile(cloud_save + "line_cloud_label.pcd", *cloud_label);
    ros::shutdown();
    return 0;
}