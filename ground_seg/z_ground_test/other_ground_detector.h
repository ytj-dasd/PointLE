#ifndef _OTHER_GROUND_DETECTOR
#define _OTHER_GROUND_DETECTOR


#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/progressive_morphological_filter.h>

#include "common_zs.h"

float ground_value = -0.45;
float min_ground_value = -0.82;
float max_ground_value = -1.65;

void PMFSegmentator(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_in, 
                    pcl::PointCloud<PointXYZLO>::Ptr &cloud_label) {

    cloud_label->resize(cloud_in->size());
    cloud_label->header = cloud_in->header;
    cloud_label->width = static_cast<uint32_t> (cloud_in->size());
    cloud_label->height = 1;
    cloud_label->is_dense = cloud_in->is_dense;
    cloud_label->sensor_orientation_ = cloud_in->sensor_orientation_;
    cloud_label->sensor_origin_ = cloud_in->sensor_origin_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filter(new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<int> cloud_filter_index;
    for (int i = 0; i < cloud_in->size(); ++i) {
        if (std::isnan(cloud_in->points[i].x) || std::isnan(cloud_in->points[i].y) ||
            std::isnan(cloud_in->points[i].z)){
            cloud_label->points[i] = PointXYZLO(0.f, 0.f, 0.f, 0, 0);
            continue;
        }
        if (std::isnan(cloud_in->points[i].z > max_ground_value)){
            cloud_label->points[i] = 
                PointXYZLO(cloud_in->points[i].x, cloud_in->points[i].y, cloud_in->points[i].z, 15, 0);
        }else {
            cloud_label->points[i] = 
                PointXYZLO(cloud_in->points[i].x, cloud_in->points[i].y, cloud_in->points[i].z, 0, 0);
            cloud_filter->push_back(cloud_in->points[i]);
            cloud_filter_index.push_back(i);
        }
    }

    pcl::ProgressiveMorphologicalFilter<pcl::PointXYZ> pmf;
    pmf.setInputCloud (cloud_filter);
    pmf.setCellSize(0.2);
    pmf.setBase(10);
    pmf.setMaxWindowSize (11);
    pmf.setSlope (0.1f);
    pmf.setInitialDistance (0.05f);
    pmf.setMaxDistance (1.0f);
    pcl::PointIndicesPtr ground(new pcl::PointIndices);
    pmf.extract(ground->indices);

    for (int i = 0; i < ground->indices.size(); ++i) {
        cloud_label->points[cloud_filter_index[ground->indices[i]]].label = 28;
    }
}

void ransacPlaneSegmentatorPrivate(pcl::PointCloud<PointXYZLO>::Ptr &cloud_label){

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ground(new pcl::PointCloud<pcl::PointXYZ>);
    for (int i = 0; i < cloud_label->size(); ++i) {
        if (cloud_label->points[i].label == 0) {
            pcl::PointXYZ pt;
            pt.x = cloud_label->points[i].x;
            pt.y = cloud_label->points[i].y;
            pt.z = cloud_label->points[i].z;
            cloud_ground->points.emplace_back(pt);
        }
    }

    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.02);
    seg.setInputCloud(cloud_ground);
    pcl::PointIndices::Ptr plane_inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr plane_model_coeff(new pcl::ModelCoefficients);
    seg.segment(*plane_inliers, *plane_model_coeff);

    if (plane_inliers->indices.size() < 100 || std::fabs(plane_model_coeff->values[2] < 0.7)) {
        std::cout << "ground plane inlier num less than: " << 200  << "or is not ground"<< std::endl;
        return;
    }
    // std::cout << "first: "
    //           << plane_model_coeff->values[0] << " " 
    //           << plane_model_coeff->values[1] << " " 
    //           << plane_model_coeff->values[2] << std::endl;

    
    for (int i = 0; i < cloud_label->size(); ++i) {
        const PointXYZLO& pt = cloud_label->points[i];
        float dist = pt.x * plane_model_coeff->values[0] + 
                     pt.y * plane_model_coeff->values[1] + 
                     pt.z * plane_model_coeff->values[2] + 
                     plane_model_coeff->values[3];
        if (fabs(dist) > 0.05) {
            continue;
        }
        cloud_label->points[i].label = 28;
    }
}

void ransacPlaneSegmentator(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_in, 
                            pcl::PointCloud<PointXYZLO>::Ptr &cloud_label){

    cloud_label->resize(cloud_in->size());
    cloud_label->header = cloud_in->header;
    cloud_label->width = static_cast<uint32_t> (cloud_in->size());
    cloud_label->height = 1;
    cloud_label->is_dense = cloud_in->is_dense;
    cloud_label->sensor_orientation_ = cloud_in->sensor_orientation_;
    cloud_label->sensor_origin_ = cloud_in->sensor_origin_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filter(new pcl::PointCloud<pcl::PointXYZ>);
    for (int i = 0; i < cloud_in->size(); ++i) {
        if (std::isnan(cloud_in->points[i].x) || std::isnan(cloud_in->points[i].y) ||
            std::isnan(cloud_in->points[i].z)){
            cloud_label->points[i] = PointXYZLO(0.f, 0.f, 0.f, 15, 0);
            continue;
        }
        if (std::isnan(cloud_in->points[i].z > max_ground_value)){
            cloud_label->points[i] = 
                PointXYZLO(cloud_in->points[i].x, cloud_in->points[i].y, cloud_in->points[i].z, 15, 0);
        }else {
            cloud_label->points[i] = 
                PointXYZLO(cloud_in->points[i].x, cloud_in->points[i].y, cloud_in->points[i].z, 0, 0);
            cloud_filter->push_back(cloud_in->points[i]);
        }
    }

    ransacPlaneSegmentatorPrivate(cloud_label);
    ransacPlaneSegmentatorPrivate(cloud_label);
    ransacPlaneSegmentatorPrivate(cloud_label);
}

Eigen::Vector3d computeNormal(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    std::vector<int> pt_index;
    for (int i = 0; i < cloud->size(); ++i) {
        if (std::isnan(cloud->points[i].x) || std::isnan(cloud->points[i].y)) {
            continue;
        }
        pt_index.push_back(i);
    }
    Eigen::Matrix<double, 3, -1> mat(3, pt_index.size());
    for (int i = 0; i < pt_index.size(); ++i) {
        Eigen::Vector3d pt(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z);
        mat.col(i) = pt.transpose();
    }
    if (mat.cols() < 3) {
        return Eigen::Vector3d(0, 0, 1);
    }
    Eigen::Vector3d pt_cnt = mat.rowwise().mean();
    Eigen::Matrix<double, 3, -1> mat_decenter =  mat.colwise() - pt_cnt;
    Eigen::Matrix<double, 3, 3> H_matrix = mat_decenter * mat_decenter.transpose();
    Eigen::JacobiSVD<Eigen::Matrix<double, 3, 3> > svd(
        H_matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix<double, 3, 3> svd_v = svd.matrixV();
    Eigen::Vector3d normal = svd_v.col(2);
    if (normal(2) < 0) {
        normal *= -1;
    }
    return normal;
}

inline bool isPoint(pcl::PointXYZ& pt) {
    if (pt.y > 2.0|| pt.y < -2.0) {
        return false;
    }
    return true;
}

void patchSegmentator(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_in,
                      pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_out) {
    int rows = cloud_in->height;
    int cols = cloud_in->width;
    cv::Mat feature_map(rows, cols, CV_8UC1, cv::Scalar(0));
    Eigen::Vector3d ground_normal(0.0278009, -0.0130179, 0.999529);
    std::cout << "patch start" << std::endl;
    for (int i = 0; i < rows - 1; ++i) {
        for (int j = 0; j < cols - 10; j++) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_patch(
                new pcl::PointCloud<pcl::PointXYZ>);
            pcl::PointXYZ pt_1 = cloud_in->at(j, i);
            pcl::PointXYZ pt_2 = cloud_in->at(j, i + 1);
            if (!isPoint(pt_1) || !isPoint(pt_2)) {
                continue;
            }
            if (std::isnan(pt_1.x) || std::isnan(pt_2.y)) {
                feature_map.at<char>(i, j) = 255;
            }
            Eigen::Vector3d vector_pt(pt_2.x - pt_1.x, pt_2.y - pt_1.y, pt_2.z - pt_1.z);
            double theta = std::acos(vector_pt.dot(ground_normal) / 
                                     vector_pt.norm());
            int feature_value = std::fabs(M_PI_2 - theta) / M_PI_2 * 250;
            feature_map.at<char>(i, j) = feature_value;
            if (std::fabs(M_PI_2 - theta) > 60 / 180 * M_PI) {
                continue;
            }
            cloud_out->push_back(pt_1);
        }
    }
    cv::imwrite("/home/tongji/tonglu/tergeo2.0/data/low_obstacle/patch_feature.jpg",
                feature_map);
    std::cout << "patch finish" << std::endl;
}

#endif