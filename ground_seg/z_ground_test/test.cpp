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
    unsigned int label_true_non_ground_num = 0;
    if (cloud_label->size() != cloud_true->size() || cloud_label->size() == 0) {
        return "-1,-1";
    }
    for (int i = 0; i < cloud_label->size(); ++i) {
        int tp_num = 0;
        int tn_num = 0;
        if (cloud_label->points[i].label == 28) {
            ++label_ground_num;
            ++tp_num;
        }else {
            ++tn_num;
        }
        if (cloud_true->points[i].label == 28) {
            ++true_ground_num;
            ++tp_num;
        }else{
            ++tn_num;
        }
        
        if (tp_num == 2) {
            ++label_true_ground_num;
        }

        if (tn_num == 2) {
            ++label_true_non_ground_num;
        }

    }

    if (label_ground_num == 0 || true_ground_num == 0) {
        return "-2,-2,-2";
    }
    double precision = (double)label_true_ground_num / (double)label_ground_num;
    double recall    = (double)label_true_ground_num / (double)true_ground_num;
    double ac        = (double)(label_true_ground_num + label_true_non_ground_num) /
        cloud_label->size();
    std::stringstream result;
    result << precision << "," << recall << "," << ac;
    return result.str();
}

bool getFiles(std::string path, std::vector<std::string>& file_name) {
    DIR *dir;
    struct dirent *ptr;
    char base[1000];
    dir = opendir(path.c_str());
    if (!dir) {
        std::cout << "can't open " << path << std::endl;
        return false;
    }
    while ( (ptr = readdir(dir)) != NULL) {
        if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0) {
            continue;
        } else if (ptr->d_type == 8) {       //file
            file_name.push_back(ptr->d_name);
        } 
        // else if (ptr->d_type == 10) {       //link_file
        //     continue;
        // } else if (ptr->d_type == 4) {        //dir
        //    continue;
        // }
    }
    closedir(dir);
    return true;
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

    std::vector<std::string> cloud_origin_file;
    if (!getFiles(cloud_origin_path, cloud_origin_file)) {
        std::cout << cloud_origin_path << " do not contain any files" << std::endl;
        return -2;
    }

    std::ofstream result_file; 
    result_file.open(result_save_path, std::ios::out | std::ios::trunc);
    if (!result_file.is_open()) {
        ROS_INFO("result_file can't open: %s", (result_save_path + ".txt").data());
    }
    result_file << "cloud_name," 
                << "mm_pre," << "mm_re," << "mm_ac,"
                << "line_pre," << "line_re," << "line_ac,"
                << "gp_pre," << "gp_re," << "gp_ac,"
                << "pmf_pre," << "pmf_re," << "pmf_ac,"
                << "rans_pre," << "rans_re," << "rans_ac" << "\n";
    
    std::cout << "cloud file size: " << cloud_origin_file.size() << std::endl;
    ros::init(argc, argv, "ground_seg_test");
    ros::NodeHandle node("~");
    // ros::NodeHandle private_node("~");
    for (int i = 0; i < cloud_origin_file.size(); ++i) {
        std::cout << "eval: " << i << " ";
        int nps = cloud_origin_file[i].find_last_of(".");
        if (nps == -1) {
            std::cout << std::endl;
            continue;
        }
        std::string cloud_type = cloud_origin_file[i].substr(nps + 1, 3);
        std::string cloud_name = cloud_origin_file[i].substr(0, nps);
        std:: cout << cloud_name << " ";
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_origin(new pcl::PointCloud<pcl::PointXYZ>);
        int load_label = 0;
        int load_true = 0;
        if (cloud_type == "ply") {
            load_label = pcl::io::loadPLYFile(cloud_origin_path + cloud_origin_file[i], *cloud_origin);
        }else if(cloud_type == "pcd") {
            load_label = pcl::io::loadPCDFile(cloud_origin_path + cloud_origin_file[i], *cloud_origin);
        }else {
            std:: cout << "type_false " << std::endl;
            continue;
        }
        pcl::PointCloud<PointXYZLO>::Ptr cloud_true(new pcl::PointCloud<PointXYZLO>);
        std::string cloud_true_name = cloud_true_path + "/" + cloud_name + "_gt.pcd";
        load_true = pcl::io::loadPCDFile(cloud_true_name, *cloud_true);
        if (load_label == -1 || load_true == -1) {
            std:: cout << "load_failed " << std::endl;;
            continue;
        }
        std::cout << std::endl;

        result_file << cloud_name << ",";
        mm::MMSegNode mm_seg_node(node);
        pcl::PointCloud<PointXYZLO>::Ptr mm_cloud_label(new pcl::PointCloud<PointXYZLO>);
        mm_seg_node.groundSeg(cloud_origin, *mm_cloud_label);
        result_file << getAccuracyAndRecall(mm_cloud_label, cloud_true) << ",";

        line::LineSegNode line_seg_node(node);
        pcl::PointCloud<PointXYZLO>::Ptr line_cloud_label(new pcl::PointCloud<PointXYZLO>);
        line_seg_node.groundSeg(cloud_origin, *line_cloud_label);
        result_file << getAccuracyAndRecall(line_cloud_label, cloud_true) << ",";

        GroundExtraction gp_ground_extraction(node, node);
        pcl::PointCloud<PointXYZLO>::Ptr gp_cloud_label(new pcl::PointCloud<PointXYZLO>);
        gp_ground_extraction.groundExtract(*cloud_origin, *gp_cloud_label);
        result_file << getAccuracyAndRecall(gp_cloud_label, cloud_true) << ",";

        pcl::PointCloud<PointXYZLO>::Ptr pmf_cloud_label(new pcl::PointCloud<PointXYZLO>);
        PMFSegmentator(cloud_origin, pmf_cloud_label);
        result_file << getAccuracyAndRecall(pmf_cloud_label, cloud_true) << ",";

        pcl::PointCloud<PointXYZLO>::Ptr rans_cloud_label(new pcl::PointCloud<PointXYZLO>);
        ransacPlaneSegmentator(cloud_origin, rans_cloud_label);
        result_file << getAccuracyAndRecall(rans_cloud_label, cloud_true) << ",";

        result_file << "\n";
    }
    
    result_file.close();
    return 0;
}