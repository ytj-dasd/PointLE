#include <ros/ros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <chrono>
#include <boost/program_options.hpp>

#include "common_zs.h"
#include "line_seg_node.h"
#include "mm_seg_node.h"
#include "ground_extraction/GroundExtraction.h"
#include "other_ground_detector.h"


void removeGround(const pcl::PointCloud<PointXYZLO>::Ptr& cloud_label, 
                  const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    for (size_t i = 0; i < cloud_label->size(); ++i) {
        if (cloud_label->points[i].label == 28) {
            continue;
        }
        cloud->points.push_back(pcl::PointXYZ(cloud_label->points[i].x, 
            cloud_label->points[i].y, cloud_label->points[i].z));
    }
}


int main(int argc, char** argv) {
    std::string cloud_path = "";
    std::string cloud_save = "";
    boost::program_options::options_description options;
    boost::program_options::variables_map var_map;
    options.add_options()("help,h", "show help");
    options.add_options()("cloud,c", boost::program_options::value<std::string>(&cloud_path), 
                         "the cloud path");
    options.add_options()("save,s", boost::program_options::value<std::string>(&cloud_save),
                          "the save path");
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, options), var_map);
    boost::program_options::notify(var_map);
    if (var_map.count("help")) {
        std::cout << options << std::endl;
        return -1;
    }
    
    ROS_INFO("cloud_path: %s", cloud_path.data());
    ROS_INFO("cloud_save: %s", cloud_save.data());

    ros::init(argc, argv, "ground_seg_test");
    ros::NodeHandle node("~");
    ros::NodeHandle private_node("~");

    int name_start = cloud_path.find_last_of("/") + 1;
    int name_end = cloud_path.find_last_of(".");
    std::string cloud_type = cloud_path.substr(name_end + 1, 3);
    std::string cloud_name = cloud_path.substr(name_start, name_end - name_start);
    cloud_save = cloud_save + "/" + cloud_name + "_";
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (cloud_type == "ply") {
        pcl::io::loadPLYFile(cloud_path, *cloud);
    }else if(cloud_type == "pcd") {
        pcl::io::loadPCDFile(cloud_path, *cloud);
    }else {
        ROS_INFO("\033[1;32m cloud type is not \033[0m");
        return 0;
    }

    ROS_INFO("\033[1;32m cloud size: %d \033[0m", cloud->size());
    std::chrono::time_point<std::chrono::steady_clock> start;
    std::chrono::time_point<std::chrono::steady_clock> end;
    std::chrono::duration<double, std::milli> dur;

    // //mm segmentation
    // ROS_INFO("\033[1;32m mm start...\033[0m");
    mm::MMSegNode mm_seg_node(node);
    pcl::PointCloud<PointXYZLO>::Ptr mm_cloud_label(new pcl::PointCloud<PointXYZLO>);
    start = std::chrono::steady_clock::now();
    mm_seg_node.groundSeg(cloud, *mm_cloud_label);
    end = std::chrono::steady_clock::now();
    dur = end - start;
    ROS_INFO("\033[1;32m MMSegNode run time: %f ms\033[0m", dur.count());
    pcl::io::savePCDFile(cloud_save + "mm_cloud_label.pcd", *mm_cloud_label);


    // //line segmentation
    // ROS_INFO("\033[1;32m line start...\033[0m");
    line::LineSegNode line_seg_node(node);
    pcl::PointCloud<PointXYZLO>::Ptr line_cloud_label(new pcl::PointCloud<PointXYZLO>);
    start = std::chrono::steady_clock::now();
    line_seg_node.groundSeg(cloud, *line_cloud_label);
    end = std::chrono::steady_clock::now();
    dur = end - start;
    ROS_INFO("\033[1;32m LineSegNode run time: %f ms\033[0m", dur.count());
    pcl::io::savePCDFile(cloud_save + "line_cloud_label.pcd", *line_cloud_label);

    //gp segmentation
    // ROS_INFO("\033[1;32m gp start...\033[0m");
    GroundExtraction gp_ground_extraction(node, private_node);
    pcl::PointCloud<PointXYZLO>::Ptr gp_cloud_label(new pcl::PointCloud<PointXYZLO>);
    start = std::chrono::steady_clock::now();
    gp_ground_extraction.groundExtract(*cloud, *gp_cloud_label);
    end = std::chrono::steady_clock::now();
    dur = end - start;
    ROS_INFO("\033[1;32mgp run time: %f ms\033[0m", dur.count());
    pcl::io::savePCDFile(cloud_save + "gp_cloud_label.pcd", *gp_cloud_label);

    
    //pmf_segmentation
    // ROS_INFO("\033[1;32m pmf_segmentation start...\033[0m");
    pcl::PointCloud<PointXYZLO>::Ptr pmf_cloud_label(new pcl::PointCloud<PointXYZLO>);
    start = std::chrono::steady_clock::now();
    PMFSegmentator(cloud, pmf_cloud_label);
    end = std::chrono::steady_clock::now();
    dur = end - start;
    ROS_INFO("\033[1;32m pmf_segmentation run time: %f ms\033[0m", dur.count());
    pcl::io::savePCDFile(cloud_save + "pmf_cloud_label.pcd", *pmf_cloud_label);
    // pcl::io::loadPCDFile("/home/zs/Downloads/people1_t_pmf_cloud_label.pcd", *pmf_cloud_label);
    // pcl::PointCloud<pcl::PointXYZ>::Ptr pmf_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    // removeGround(pmf_cloud_label, pmf_cloud);
    // pcl::io::savePLYFile(cloud_save + "pmf_cloud.ply", *pmf_cloud);

    //ransacPlaneSegmentator
    // ROS_INFO("\033[1;32m ransacPlaneSegmentator start...\033[0m");
    start = std::chrono::steady_clock::now();
    pcl::PointCloud<PointXYZLO>::Ptr rans_cloud_label(new pcl::PointCloud<PointXYZLO>);
    ransacPlaneSegmentator(cloud, rans_cloud_label);
    end = std::chrono::steady_clock::now();
    dur = end - start;
    ROS_INFO("\033[1;32m ransacPlaneSegmentator run time: %f ms\033[0m", dur.count());
    pcl::io::savePCDFile(cloud_save + "rans_cloud_label.pcd", *rans_cloud_label);

    // //patchSegmentator
    // ROS_INFO("\033[1;32m patchSegmentator start...\033[0m");
    // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ground_patch(new pcl::PointCloud<pcl::PointXYZ>);
    // start = std::chrono::steady_clock::now();
    // patchSegmentator(cloud, cloud_ground_patch);
    // end = std::chrono::steady_clock::now();
    // dur = end - start;
    // ROS_INFO("\033[1;32m patchSegmentator run time: %f ms\033[0m", dur.count());
    // // pcl::io::savePLYFile(cloud_save + "cloud_ground_patch.ply", *cloud_ground_patch);
    // pcl::io::savePCDFile(cloud_save + "cloud_ground_patch.pcd", *cloud_ground_patch);
    ros::shutdown();
    return 0;
}