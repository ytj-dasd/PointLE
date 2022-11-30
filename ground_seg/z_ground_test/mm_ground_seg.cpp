#include <chrono>
#include <fstream>
#include <sstream>
#include <ros/ros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <boost/program_options.hpp>
#include <pcl/filters/filter.h>

#include "common_zs.h"
#include "mm_seg_node.h"
#include "line_seg_node.h"
#include "ground_extraction/GroundExtraction.h"

struct PCD
{
  PointCloud::Ptr cloud;
  std::string f_name;

  PCD() : cloud (new PointCloud) {};
};

void loadData ( std::vector<PCD, Eigen::aligned_allocator<PCD> > &models)
{
  struct dirent **name_list;
    pcl::PCDReader reader;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    std::string  path= "/home/ytj/文档/ground_seg/src/z_ground_test/car_raw/";
      //  std::string path = "/home/ytj/Documents/Deep_Learning/data/pcd/result/";

    int n = scandir(path.data(), &name_list, 0, versionsort);
    std::cout<<"size of PCD:"<<n-2<<std::endl;
    if (n < 0)
    {
        printf("scandir return %d \n", n);
    }
    else
    {
        int index = 2;
        while (index <= n - 1)
        {
            std::string name = name_list[index]->d_name;
            // std::cout<<name<<std::endl;
            PCD m;
            m.f_name = name;
            reader.read(path + name, *m.cloud);
            std::vector<int> indices;
            pcl::removeNaNFromPointCloud(*m.cloud,*m.cloud, indices);
            models.push_back (m);
            free(name_list[index++]);
        }
        free(name_list);
    }
}

int main(int argc, char** argv) {
    //Input Output路径加载
    // std::string cloud_origin_path = "";
    // std::string result_save_path  = "";
    // boost::program_options::options_description options;
    // boost::program_options::variables_map var_map;
    // options.add_options()("help,h", "show help");
    // options.add_options()("origin,o", boost::program_options::value<std::string>(&cloud_origin_path), 
    //                      "the cloud origin path");
    // options.add_options()("save,s", boost::program_options::value<std::string>(&result_save_path),
    //                     "the result save path");
    // boost::program_options::store(boost::program_options::parse_command_line(argc, argv, options), var_map);
    // boost::program_options::notify(var_map);
    // if (var_map.count("help")) {
    //     std::cout << options << std::endl;
    //     return -1;
    // }
    
    // ROS_INFO("cloud_origin_path: %s", cloud_origin_path.data());
    // ROS_INFO("result_save_path: %s" , result_save_path.data());

    ros::init(argc, argv, "ground_seg_extract");
    ros::NodeHandle node("~");

    // int name_start = cloud_origin_path.find_last_of("/") + 1;
    // int name_end = cloud_origin_path.find_last_of(".");
    // std::string cloud_type = cloud_origin_path.substr(name_end + 1, 3);
    // std::string cloud_name = cloud_origin_path.substr(name_start, name_end - name_start);

    std::vector<PCD, Eigen::aligned_allocator<PCD> > data;
    loadData (data);
    
    std::stringstream s;
    s << "/home/ytj/文档/pre_process/DataLoad/vehicle_1/"<< "car2_noground.txt";
    std::ofstream outfile(s.str(), std::ios::app);
    for (std::size_t i = 0; i < data.size (); i++)
    {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_origin(new pcl::PointCloud<pcl::PointXYZ>);
    cloud_origin = data[i].cloud;

    mm::MMSegNode mm_seg_node(node);
    pcl::PointCloud<PointXYZLO>::Ptr mm_cloud_label(new pcl::PointCloud<PointXYZLO>);
    mm_seg_node.groundSeg(cloud_origin, *mm_cloud_label);
    // pcl::io::savePCDFile(result_save_path + "/" + cloud_name + "_mm_ground.pcd" ,*mm_cloud_label);

    // //line segmentation
    // line::LineSegNode line_seg_node(node);
    // pcl::PointCloud<PointXYZLO>::Ptr line_cloud_label(new pcl::PointCloud<PointXYZLO>);
    // line_seg_node.groundSeg(cloud_origin, *line_cloud_label);
    // pcl::io::savePCDFile(result_save_path + "/" + cloud_name + "_line_ground.pcd" ,*line_cloud_label);

    // //gp segmentation
    // GroundExtraction gp_ground_extraction(node, node);
    // pcl::PointCloud<PointXYZLO>::Ptr gp_cloud_label(new pcl::PointCloud<PointXYZLO>);
    // gp_ground_extraction.groundExtract(*cloud_origin, *gp_cloud_label);
    // pcl::io::savePCDFile(result_save_path + "/" + cloud_name + "_gp_ground.pcd" ,*gp_cloud_label);
    // return 0;

    //seg
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_no_ground(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ground(new pcl::PointCloud<pcl::PointXYZ>);
    // cloud_no_ground->points.reserve(cloud_origin->size());
    
    for (int j = 0; j < mm_cloud_label->size(); j++) {
        if (mm_cloud_label->points[j].label == 28) 
        {
            cloud_ground->points.push_back(cloud_origin->points[j]);
        }
        cloud_no_ground->points.push_back(cloud_origin->points[j]);
    }
    // cloud_no_ground->height = 1;
    // cloud_no_ground->width = cloud_no_ground->size();
    // std::cout<<cloud_no_ground->points[0].x << " "<<cloud_no_ground->points[0].y << " "<<cloud_no_ground->points[0].z << " "<<std::endl;
    pcl::io::savePCDFile("/home/ytj/文档/ground_seg/src/z_ground_test/car_result/noground_" + data[i].f_name ,*cloud_no_ground,true);
    pcl::io::savePCDFile("/home/ytj/文档/ground_seg/src/z_ground_test/car_result/ground_" + data[i].f_name ,*cloud_ground,true);
    if(cloud_no_ground->size()< 22) std::cout<<data[i].f_name<<" "<<cloud_no_ground->size()<<std::endl;
    outfile << cloud_no_ground->size() <<std::endl;
    }
    // outfile.close();
}