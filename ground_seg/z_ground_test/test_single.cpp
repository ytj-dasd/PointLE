#include <chrono>
#include <fstream>
#include <sstream>
#include <ros/ros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <boost/program_options.hpp>

#include "common_zs.h"
#include "line_seg_node.h"
#include "mm_seg_node.h"
#include "ground_extraction/GroundExtraction.h"
#include "other_ground_detector.h"

std::string getAccuracyAndRecall(const pcl::PointCloud<PointXYZLO>::Ptr& cloud_label,
                                 const pcl::PointCloud<PointXYZLO>::Ptr& cloud_true) {
    unsigned int true_ground_num = 0;
    unsigned int label_ground_num = 0;
    unsigned int label_true_ground_num = 0;
    if (cloud_label->size() != cloud_true->size() || cloud_label->size() == 0) {
        return "-1,-1";
    }
    for (int i = 0; i < cloud_label->size(); ++i) {
        int temp = 0;
        if (cloud_label->points[i].label == 28) {
            ++label_ground_num;
            ++temp;
        }
        if (cloud_true->points[i].label == 28) {
            ++true_ground_num;
            ++temp;
        }
        if (temp == 2) {
            ++label_true_ground_num;
        }
    }

    if (label_ground_num == 0 || true_ground_num == 0) {
        return "-2,-2";
    }
    double accuracy = (double)label_true_ground_num / (double)label_ground_num;
    double recall   = (double)label_true_ground_num / (double)true_ground_num;
    std::stringstream result;
    result << "ac:  " << accuracy << " re: " << recall;
    return result.str();
}

int main(int argc, char** argv) {
    std::string cloud_origin_path = "";
    std::string cloud_true_path   = "";
    std::string result_save_path  = "";
    boost::program_options::options_description options;
    boost::program_options::variables_map var_map;
    options.add_options()("help,h", "show help");
    options.add_options()("origin,o", boost::program_options::value<std::string>(&cloud_origin_path), 
                         "the cloud origin path");
    options.add_options()("true,t", boost::program_options::value<std::string>(&cloud_true_path),
                          "the cloud true path");
    options.add_options()("save,s", boost::program_options::value<std::string>(&result_save_path),
                        "the result save path");
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, options), var_map);
    boost::program_options::notify(var_map);
    if (var_map.count("help")) {
        std::cout << options << std::endl;
        return -1;
    }
    
    ROS_INFO("cloud_origin_path: %s", cloud_origin_path.data());
    ROS_INFO("cloud_true_path: %s"  , cloud_true_path.data());
    ROS_INFO("result_save_path: %s" , result_save_path.data());

    ros::init(argc, argv, "ground_seg_test_single");
    ros::NodeHandle node("~");
    ros::NodeHandle private_node("~");


    int name_start = cloud_origin_path.find_last_of("/") + 1;
    int name_end = cloud_origin_path.find_last_of(".");
    std::string cloud_type = cloud_origin_path.substr(name_end + 1, 3);
    std::string cloud_name = cloud_origin_path.substr(name_start, name_end - name_start);
    // cloud_true_path = cloud_true_path + "/" + cloud_name + "_gt";
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_origin(new pcl::PointCloud<pcl::PointXYZ>);
    if (cloud_type == "ply") {
        pcl::io::loadPLYFile(cloud_origin_path, *cloud_origin);
    }else if(cloud_type == "pcd") {
        pcl::io::loadPCDFile(cloud_origin_path, *cloud_origin);
    }else {
        ROS_INFO("\033[1;32m cloud type is not \033[0m");
        return 0;
    }
    pcl::PointCloud<PointXYZLO>::Ptr cloud_true(new pcl::PointCloud<PointXYZLO>);
    pcl::io::loadPCDFile(cloud_true_path, *cloud_true);

    std::stringstream result;
    result << "cloud_name: " << cloud_name << "\n";

    mm::MMSegNode mm_seg_node(node);
    pcl::PointCloud<PointXYZLO>::Ptr mm_cloud_label(new pcl::PointCloud<PointXYZLO>);
    mm_seg_node.groundSeg(cloud_origin, *mm_cloud_label);
    result << "mm:   "<< getAccuracyAndRecall(mm_cloud_label, cloud_true) << "\n";

    line::LineSegNode line_seg_node(node);
    pcl::PointCloud<PointXYZLO>::Ptr line_cloud_label(new pcl::PointCloud<PointXYZLO>);
    line_seg_node.groundSeg(cloud_origin, *line_cloud_label);
    result << "line: "<< getAccuracyAndRecall(line_cloud_label, cloud_true) << "\n";

    GroundExtraction gp_ground_extraction(node, private_node);
    pcl::PointCloud<PointXYZLO>::Ptr gp_cloud_label(new pcl::PointCloud<PointXYZLO>);
    gp_ground_extraction.groundExtract(*cloud_origin, *gp_cloud_label);
    result << "gp:   "<< getAccuracyAndRecall(gp_cloud_label, cloud_true) << "\n";

    pcl::PointCloud<PointXYZLO>::Ptr pmf_cloud_label(new pcl::PointCloud<PointXYZLO>);
    PMFSegmentator(cloud_origin, pmf_cloud_label);
    result << "pmf:  "<< getAccuracyAndRecall(pmf_cloud_label, cloud_true) << "\n";

    pcl::PointCloud<PointXYZLO>::Ptr rans_cloud_label(new pcl::PointCloud<PointXYZLO>);
    ransacPlaneSegmentator(cloud_origin, rans_cloud_label);
    result << "rans: "<< getAccuracyAndRecall(rans_cloud_label, cloud_true) << "\n";

    std::cout << std::endl;
    std::cout << "----------------------------------------------------" << std::endl;
    std::cout << result.str() << std::endl;
    std::cout << "----------------------------------------------------" << std::endl;
    return 0;
}