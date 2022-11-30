#include "line_seg_node.h"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/ply_io.h>
namespace line{
LineSegNode::LineSegNode(ros::NodeHandle& node){
  
    std::string lidar_topic_name = getLidarTopicName(node);
    std::string ground_topic_name = getGroundTopicName(node);
    std::string obstacle_topic_name = getObstacleTopicName(node);
    getGroundSegmentationParams(node);
    
    _lidar_sub = node.subscribe(lidar_topic_name, 10, &LineSegNode::pointCloudCallback, this);
    _ground_pub = node.advertise<sensor_msgs::PointCloud2>(ground_topic_name, 10); 
    _obstacle_pub = node.advertise<sensor_msgs::PointCloud2>(obstacle_topic_name, 10); 
}

LineSegNode::~LineSegNode(){

}

std::string LineSegNode::getLidarTopicName(ros::NodeHandle& node) {
    std::string lidar_topic_name;
    if (!node.getParam("lidar_topic_name", lidar_topic_name)) {
        lidar_topic_name = "";
    }
    return lidar_topic_name;
}

std::string LineSegNode::getGroundTopicName(ros::NodeHandle& node){
    std::string ground_topic_name;
    if (!node.getParam("line_ground_topic_name", ground_topic_name)) {
        ground_topic_name = "";
    }
    return ground_topic_name;
}

std::string LineSegNode::getObstacleTopicName(ros::NodeHandle& node){
    std::string obstalce_topic_name;
    if (!node.getParam("line_obstalce_topic_name", obstalce_topic_name)) {
        obstalce_topic_name = "";
    }
    return obstalce_topic_name;
}

void LineSegNode::getGroundSegmentationParams(ros::NodeHandle& node) {
    GroundSegmentationParams params;
    node.param("line/n_threads", params.n_threads, params.n_threads);
    node.param("line/r_min_square", params.r_min_square, params.r_min_square);
    node.param("line/r_max_square", params.r_max_square, params.r_max_square);
    node.param("line/n_bins", params.n_bins, params.n_bins);
    node.param("line/n_segments", params.n_segments, params.n_segments);
    node.param("line/max_dist_to_line", params.max_dist_to_line, params.max_dist_to_line);
    node.param("line/sensor_height", params.sensor_height, params.sensor_height);
    node.param("line/max_slope", params.max_slope, params.max_slope);
    node.param("line/max_fit_error_square", params.max_error_square, params.max_error_square);
    node.param("line/long_threshold", params.long_threshold, params.long_threshold);
    node.param("line/max_long_height", params.max_long_height, params.max_long_height);
    node.param("line/max_start_height", params.max_start_height, params.max_start_height);
    node.param("line/line_search_angle", params.line_search_angle, params.line_search_angle);
    node.param("line/visualize", params.visualize, params.visualize);
    // node.param("ground_max_height", params.ground_max_height, params.ground_max_height);
    // _ground_segmentator.setParams(params);
    _ground_segmentator_params = params;
    node.param("line/ground_value", _ground_value, _ground_value);
    _filter_heigh = 3.8 - params.sensor_height;
    _filter_heigh_low = -3.8 - params.sensor_height;
}

void LineSegNode::processCloud(const sensor_msgs::PointCloud2& cloud_msg,
    pcl::PointCloud<pcl::PointXYZ>& cloud_out) {

    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::fromROSMsg(cloud_msg, cloud);
    cloud_out.clear();
    for (int i = 0; i < cloud.size(); ++i) {
        if (std::isnan(cloud.points[i].x) || std::isnan(cloud.points[i].y) || 
           std::isnan(cloud.points[i].z) || cloud.points[i].z > _filter_heigh) {
           continue;
        }
        cloud_out.points.emplace_back(cloud.points[i]);
    }
}

void LineSegNode::pointCloudCallback(const sensor_msgs::PointCloud2& cloud_msg){
    
    ++_frame_num;
    _frame_num %= 100000007;
    if(_frame_num % 3 != 0) {
        return;
    }

    //process input cloud msg
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(cloud_msg, *cloud);
    pcl::io::savePLYFile("/home/zs/zs/kitti/data/cloud_origin.ply", *cloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr obstacle_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr ground_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    groundSeg(cloud, *obstacle_cloud, *ground_cloud);
    pcl::io::savePLYFile("/home/zs/zs/kitti/data/cloud_obstacle.ply", *obstacle_cloud);
    // std::cout << "ground_cloud " << ground_cloud.points.size() << std::endl; 
    // std::cout << "obstacle_cloud " << obstacle_cloud.points.size() << std::endl; 

    //publish ground cloud
    sensor_msgs::PointCloud2 ground_msg;
    pcl::toROSMsg(*ground_cloud, ground_msg);
    ground_msg.header.frame_id = "velo_link";
    ground_msg.header.stamp = cloud_msg.header.stamp;
    _ground_pub.publish(ground_msg);

    //publish obstacle cloud
    sensor_msgs::PointCloud2 obstacle_msg;
    pcl::toROSMsg(*obstacle_cloud, obstacle_msg);
    obstacle_msg.header.frame_id = "velo_link";
    obstacle_msg.header.stamp = cloud_msg.header.stamp;
    _obstacle_pub.publish(obstacle_msg);
}

void LineSegNode::groundSeg(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_in,
                        pcl::PointCloud<pcl::PointXYZ>& cloud_obstacle,
                        pcl::PointCloud<pcl::PointXYZ>& cloud_ground) {
    pcl::PointCloud<pcl::PointXYZ> process_cloud;
    process_cloud.clear();
    cloud_obstacle.clear();
    cloud_ground.clear();
    cloud_obstacle.reserve(cloud_in->size());
    cloud_ground.reserve(cloud_in->size());
    for (int i = 0; i < cloud_in->size(); ++i) {
        // if (std::isnan(cloud_in->points[i].x) || std::isnan(cloud_in->points[i].y) || 
        //     std::isnan(cloud_in->points[i].z) || cloud_in->points[i].z < _filter_heigh_low) {
        //     continue;
        // }
        if (cloud_in->points[i].z > _filter_heigh) {
            cloud_obstacle.points.emplace_back(cloud_in->points[i]);
        }else {
            process_cloud.points.emplace_back(cloud_in->points[i]);
        }
    }
    // std::cout << "_filter_heigh: " << _filter_heigh << std::endl;
    // pcl::io::savePLYFile("/home/zs/zs/kitti/data/line_groundSeg_process_cloud.ply", process_cloud);
    // pcl::io::savePLYFile("/home/zs/zs/kitti/data/line_groundSeg_cloud_obstacle.ply", cloud_obstacle);
    std::vector<int> labels;
    GroundSegmentation ground_segmentator(_ground_segmentator_params);
    ground_segmentator.segment(process_cloud, &labels);
    pcl::PointCloud<pcl::PointXYZ> ground_cloud;
    cloud_ground.reserve(process_cloud.size());
    for (int i = 0; i < process_cloud.size(); ++i){
        if (labels[i] == 1) {
            cloud_ground.points.emplace_back(process_cloud.points[i]);
        }else {
            cloud_obstacle.points.emplace_back(process_cloud.points[i]);
        }
    }
    cloud_ground.points.shrink_to_fit();
    cloud_obstacle.points.shrink_to_fit();
}

void LineSegNode::groundSeg(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_in,
                            pcl::PointCloud<PointXYZLO>& cloud_label){

    pcl::PointCloud<pcl::PointXYZ> process_cloud;
    std::vector<int> process_cloud_index;
    cloud_label.header = cloud_in->header;
    cloud_label.width = static_cast<uint32_t> (cloud_in->size ());
    cloud_label.height = 1;
    cloud_label.is_dense = cloud_in->is_dense;
    cloud_label.sensor_orientation_ = cloud_in->sensor_orientation_;
    cloud_label.sensor_origin_ = cloud_in->sensor_origin_;
    for (int i = 0; i < cloud_in->size(); ++i) {
        if (std::isnan(cloud_in->points[i].x) || std::isnan(cloud_in->points[i].y) || 
            std::isnan(cloud_in->points[i].z) ) {
            
            cloud_label.points.emplace_back(PointXYZLO(0.f, 0.f, 0.f, 0, 0));
            continue;
        }
        if (cloud_in->points[i].z > _filter_heigh) {
            cloud_label.points.emplace_back(
                PointXYZLO(cloud_in->points[i].x, cloud_in->points[i].y, cloud_in->points[i].z, 0, 0));
        }else {
            cloud_label.points.emplace_back(
                PointXYZLO(cloud_in->points[i].x, cloud_in->points[i].y, cloud_in->points[i].z, 0, 0));
            process_cloud.points.emplace_back(cloud_in->points[i]);
            process_cloud_index.emplace_back(i);
        }
    }
    // std::cout << "_filter_heigh: " << _filter_heigh << std::endl;
    // pcl::io::savePLYFile("/home/zs/zs/kitti/data/line_groundSeg_process_cloud.ply", process_cloud);
    // pcl::io::savePLYFile("/home/zs/zs/kitti/data/line_groundSeg_cloud_obstacle.ply", cloud_obstacle);
    std::vector<int> labels;
    GroundSegmentation ground_segmentator(_ground_segmentator_params);
    ground_segmentator.segment(process_cloud, &labels);
    for (int i = 0; i < process_cloud.size(); ++i){
        if (labels[i] == 1) {
            cloud_label.points[process_cloud_index[i]].label = 28;
        }
    }
}
}