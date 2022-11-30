#include "hybrid_model_regression/ground_segmentation.h"

#include <chrono>
#include <cmath>
#include <list>
#include <memory>
#include <thread>

#include "hybrid_model_regression/gp_predictor.h"
namespace mm{
MMGroundSegmentation::MMGroundSegmentation(const MMGroundSegmentationParams& seg_params,
                                       const GPModelParams& gp_model_params,
                                       const GPPreParams& gp_pre_params) : 
                                       _seg_params(seg_params),
                                       _gp_pre_params(gp_pre_params),
                                       _gp_model_params(gp_model_params){
    double bin_num = std::ceil((seg_params.r_max - seg_params.r_min) / seg_params.bin_length);
    _segments = std::vector<Segment>(seg_params.n_segments, Segment(bin_num, 
                                     seg_params.max_slope, seg_params.max_error_square, 
                                     seg_params.long_threshold, seg_params.max_long_height, 
                                     seg_params.max_start_height, seg_params.sensor_height,
                                     seg_params.bin_length));
    if (seg_params.visualize) {
        _viewer = std::make_shared<pcl::visualization::PCLVisualizer>("3D Viewer");
    }
}

void MMGroundSegmentation::segment(const PointCloud& cloud, std::vector<int>* segmentation) {

    // std::cout << "MM segmenting cloud with " << cloud.size() << " points. ";
    std::chrono::high_resolution_clock::time_point start = 
        std::chrono::high_resolution_clock::now();
    segmentation->clear();
    segmentation->resize(cloud.size(), 0);
    _bin_index.resize(cloud.size());
    _segment_coordinates.resize(cloud.size());
    //Line Model
    insertPoints(cloud);
    std::list<PointLine> lines;
    if (_seg_params.visualize) {
        getLines(&lines);
    } else {
        getLines(NULL);
    }
    assignCluster(segmentation);
    // saveLineGroundPoint("/home/zs/zs/kitti/data/ground_seg", segmentation);
    // saveSegmentPoint("/home/zs/zs/kitti/data/ground_seg/segment_point/");
    if (_seg_params.visualize) {
        // Visualize.
        PointCloud::Ptr obstacle_cloud(new PointCloud());
        // Get cloud of ground points.
        PointCloud::Ptr ground_cloud(new PointCloud());
        for (size_t i = 0; i < cloud.size(); ++i) {
            if (segmentation->at(i) == 1) {
                ground_cloud->push_back(cloud[i]);
            } else {
                obstacle_cloud->push_back(cloud[i]);
            }
        }
        PointCloud::Ptr min_cloud(new PointCloud());
        getGroundPointCloud(min_cloud.get());
        visualize(lines, min_cloud, ground_cloud, obstacle_cloud);
    }
    // return;
    // GP Model
    // normaliazation z
    std::vector<std::vector<Bin> > ground_skeletons(
        _seg_params.n_segments, std::vector<Bin>(_segments[0].size()));
    std::vector<std::vector<int> > segs_obstacle_pt_index(_seg_params.n_segments);
    // std::cout << _seg_params.n_segments << " " << _segments[0].size() << std::endl;
    for (int i = 0; i < _segment_coordinates.size(); ++i) {
        _segment_coordinates[i].z -= _seg_params.sensor_height;
        if (_bin_index[i].first == -1) {
            continue;
        }
        if(segmentation->at(i) == 1) {
            ground_skeletons[_bin_index[i].first][_bin_index[i].second].
                addGroundPoint(_segment_coordinates[i]);
        }else {
            segs_obstacle_pt_index[_bin_index[i].first].push_back(i);
        }
    }
    GPPredictor gp_predictor(_gp_model_params, _gp_pre_params);
    gp_predictor.predict(ground_skeletons, _segment_coordinates, 
                         segs_obstacle_pt_index, segmentation, true);

    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> fp_ms = end - start;
    // std::cout << "Done! Took " << fp_ms.count() << "ms\n";
}

void MMGroundSegmentation::insertPoints(const PointCloud& cloud) {
    std::vector<std::thread> threads(_seg_params.n_threads);
    const size_t points_per_thread = cloud.size() / _seg_params.n_threads;
    // Launch threads.
    for (unsigned int i = 0; i < _seg_params.n_threads - 1; ++i) {
        const size_t start_index = i * points_per_thread;
        const size_t end_index = (i + 1) * points_per_thread;
        threads[i] = std::thread(&MMGroundSegmentation::insertionThread, this,
                                 cloud, start_index, end_index);
    }
    // Launch last thread which might have more points than others.
    const size_t start_index = (_seg_params.n_threads - 1) * points_per_thread;
    const size_t end_index = cloud.size();
    threads[_seg_params.n_threads - 1] =
        std::thread(&MMGroundSegmentation::insertionThread, this, cloud, start_index, end_index);
    // Wait for threads to finish.
    for (auto it = threads.begin(); it != threads.end(); ++it) {
        if (it->joinable()) {
            it->join();
        }
    }
}

void MMGroundSegmentation::insertionThread(const PointCloud& cloud,
                                         const size_t start_index,
                                         const size_t end_index) {
    const double segment_step = 2 * M_PI / _seg_params.n_segments;
    for (unsigned int i = start_index; i < end_index; ++i) {
        pcl::PointXYZ point(cloud[i]);
        const double range = std::sqrt(point.x * point.x + point.y * point.y);
        if (range < _seg_params.r_max && range > _seg_params.r_min) {
            const double angle = std::atan2(point.y, point.x);
            const unsigned int bin_index = (range - _seg_params.r_min) / _seg_params.bin_length;
            const unsigned int segment_index = (angle + M_PI) / segment_step;
            const unsigned int segment_index_clamped = 
                segment_index == _seg_params.n_segments ? 0 : segment_index;
            _segments[segment_index_clamped][bin_index].addPoint(range, point.z);
            _bin_index[i] = std::make_pair(segment_index_clamped, bin_index);
        } else {
            _bin_index[i] = std::make_pair<int, int>(-1, -1);
        }
        _segment_coordinates[i] = GroundPoint(range, point.z);
    }
}


void MMGroundSegmentation::getLines(std::list<PointLine> *lines) {
    std::mutex line_mutex;
    std::vector<std::thread> thread_vec(_seg_params.n_threads);
    unsigned int i;
    for (i = 0; i < _seg_params.n_threads; ++i) {
        const unsigned int start_index = _seg_params.n_segments / _seg_params.n_threads * i;
        const unsigned int end_index = _seg_params.n_segments / _seg_params.n_threads * (i + 1);
        thread_vec[i] = std::thread(&MMGroundSegmentation::lineFitThread, this,
                                    start_index, end_index, lines, &line_mutex);
    }
    for (auto it = thread_vec.begin(); it != thread_vec.end(); ++it) {
        if (it->joinable()) {
            it->join();
        }
    }
}

void MMGroundSegmentation::lineFitThread(const unsigned int start_index,
                                       const unsigned int end_index,
                                       std::list<PointLine> *lines, std::mutex* lines_mutex) {
    const bool visualize = lines;
    const double seg_step = 2 * M_PI / _seg_params.n_segments;
    double angle = -M_PI + seg_step / 2 + seg_step * start_index;
    for (unsigned int i = start_index; i < end_index; ++i) {
        _segments[i].fitSegmentLines();
        // _segments[i].originFitSegmentLines();
        // Convert lines to 3d if we want to.
        if (!visualize) {
            continue;
        }
        std::list<Segment::Line> segment_lines;
        _segments[i].getLines(&segment_lines);
        for (auto line_iter = segment_lines.begin(); line_iter != segment_lines.end(); ++line_iter) {
          const pcl::PointXYZ start = GroundPointTo3d(line_iter->first, angle);
          const pcl::PointXYZ end = GroundPointTo3d(line_iter->second, angle);
          lines_mutex->lock();
          lines->emplace_back(start, end);
          lines_mutex->unlock();
        }
        angle += seg_step;
    }
}

void MMGroundSegmentation::assignCluster(std::vector<int>* segmentation) {
    std::vector<std::thread> thread_vec(_seg_params.n_threads);
    const unsigned int step_size = segmentation->size() / _seg_params.n_threads;
    for (unsigned int i = 0; i < _seg_params.n_threads; ++i) {
        const unsigned int start_index = step_size * i;
        const unsigned int end_index = step_size * (i + 1);
        thread_vec[i] = std::thread(&MMGroundSegmentation::assignClusterThread, this,
                                    start_index, end_index, segmentation);
    }
    for (auto it = thread_vec.begin(); it != thread_vec.end(); ++it) {
        if (it->joinable()) {
            it->join();
        }
    }
}

void MMGroundSegmentation::assignClusterThread(const unsigned int &start_index,
                                             const unsigned int &end_index,
                                             std::vector<int> *segmentation) {
    const double segment_step = 360 / _seg_params.n_segments;
    for (unsigned int i = start_index; i < end_index; ++i) {
        GroundPoint point_2d = _segment_coordinates[i];
        const int segment_index = _bin_index[i].first;
        if (segment_index < 0) {
            continue;
        }
        double dist = _segments[segment_index].verticalDistanceToLine(point_2d.d, point_2d.z);
        // Search neighboring segments.
        int steps = 1;
        while (dist == -1 && steps * segment_step < _seg_params.line_search_angle) {
            // Fix indices that are out of bounds.
            int index_1 = segment_index + steps;
            while (index_1 >= _seg_params.n_segments) index_1 -= _seg_params.n_segments;
            int index_2 = segment_index - steps;
            while (index_2 < 0) index_2 += _seg_params.n_segments;
            // Get distance to neighboring lines.
            const double dist_1 = _segments[index_1].verticalDistanceToLine(point_2d.d, point_2d.z);
            const double dist_2 = _segments[index_2].verticalDistanceToLine(point_2d.d, point_2d.z);
            // Select larger distance if both segments return a valid distance.
            if (dist_1 > dist) {
                dist = dist_1;
            }
            if (dist_2 > dist) {
                dist = dist_2;
            }
            ++steps;
        }
        if (dist < _seg_params.max_dist_to_line && dist != -1) {
            segmentation->at(i) = 1;
        }
    }
}

void MMGroundSegmentation::getGroundPointCloud(PointCloud* cloud) {
    const double seg_step = 2 * M_PI / _seg_params.n_segments;
    double angle = -M_PI + seg_step / 2;
    for (auto seg_iter = _segments.begin(); seg_iter != _segments.end(); ++seg_iter) {
        for (auto bin_iter = seg_iter->begin(); bin_iter != seg_iter->end(); ++bin_iter) {
            const pcl::PointXYZ min = GroundPointTo3d(bin_iter->getGroundPoint(), angle);
            cloud->push_back(min);
        }
        angle += seg_step;
    }
}

pcl::PointXYZ MMGroundSegmentation::GroundPointTo3d(const GroundPoint &min_z_point,
                                                const double &angle) {
    pcl::PointXYZ point;
    point.x = cos(angle) * min_z_point.d;
    point.y = sin(angle) * min_z_point.d;
    point.z = min_z_point.z;
    return point;
}

void MMGroundSegmentation::getGroundPoints(PointCloud* out_cloud) {
    const double seg_step = 2 * M_PI / _seg_params.n_segments;
    double angle = -M_PI + seg_step / 2;
    for (auto seg_iter = _segments.begin(); seg_iter != _segments.end(); ++seg_iter) {
        double dist = _seg_params.r_min + _seg_params.bin_length / 2;
        for (auto bin_iter = seg_iter->begin(); bin_iter != seg_iter->end(); ++bin_iter) {
            pcl::PointXYZ point;
            if (bin_iter->hasPoint()) {
                GroundPoint min_z_point(bin_iter->getGroundPoint());
                point.x = cos(angle) * min_z_point.d;
                point.y = sin(angle) * min_z_point.d;
                point.z = min_z_point.z;

                out_cloud->push_back(point);
            }
            dist += _seg_params.bin_length;
        }
        angle += seg_step;
    }
}

void MMGroundSegmentation::visualizePointCloud(const PointCloud::ConstPtr& cloud,
                                             const std::string& id) {
    _viewer->addPointCloud(cloud, id, 0);
}

void MMGroundSegmentation::visualizeLines(const std::list<PointLine>& lines) {
    size_t counter = 0;
    for (auto it = lines.begin(); it != lines.end(); ++it) {
      _viewer->addLine<pcl::PointXYZ>(it->first, it->second, std::to_string(counter++));
    }
}

void MMGroundSegmentation::visualize(const std::list<PointLine>& lines,
                                    const PointCloud::ConstPtr& min_cloud,
                                    const PointCloud::ConstPtr& ground_cloud,
                                    const PointCloud::ConstPtr& obstacle_cloud) {
    _viewer->setBackgroundColor (0, 0, 0);
    _viewer->addCoordinateSystem (1.0);
    _viewer->initCameraParameters ();
    _viewer->setCameraPosition(-2.0, 0, 2.0, 1.0, 0, 0);
    visualizePointCloud(min_cloud, "min_cloud");
    _viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR,
                                              0.0f, 1.0f, 0.0f,
                                              "min_cloud");
    _viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                              2.0f,
                                              "min_cloud");
    visualizePointCloud(ground_cloud, "ground_cloud");
    _viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR,
                                              1.0f, 0.0f, 0.0f,
                                              "ground_cloud");
    visualizePointCloud(obstacle_cloud, "obstacle_cloud");
    visualizeLines(lines);
    while (!_viewer->wasStopped ()){
        _viewer->spin();
        // boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }
}

void MMGroundSegmentation::saveLineGroundPoint(std::string file_path, const std::vector<int>* segmentation) {
    
    for (int i = 0; i < segmentation->size(); ++i) {
        if (segmentation->at(i) != 1) {
            continue;
        }
        std::ofstream file;
        file.open(file_path + "/" + std::to_string(_bin_index[i].first) + ".txt", 
            std::ios::out | std::ios::app);
        if (!file.is_open()) {
            std::cout << file_path << "/" << std::to_string(_bin_index[i].first) << ".txt" 
                      << " can't open" <<  std::endl;
            continue;
        }
        file << _segment_coordinates[i].d << " "
             << _segment_coordinates[i].z << "\n";
        file.close();
    }
}

void MMGroundSegmentation::saveSegmentPoint(std::string file_path){
    std::ofstream file;
    file.open(file_path + "/segment.txt", std::ios::out | std::ios::app);
    if (!file.is_open()) {
        std::cout << file_path << "/segment.txt" << " can't open" <<  std::endl;
        return;
    }
    for (int i = 0; i < _segments.size(); ++i) {
        int point_num = 0;
        for (int j = 0; j < _segments[i].size(); ++j) {
            if (!_segments[i][j].hasPoint()) {
                continue;
            }
            point_num++;
        }
        file << i << ": " << point_num << "\n";
    }
    file.close();
    // for (int i = 0; i < _segments.size(); ++i) {
    //     std::ofstream file;
    //     file.open(file_path + "/" + std::to_string(i) + ".txt", 
    //         std::ios::out | std::ios::app);
    //     if (!file.is_open()) {
    //         std::cout << file_path << "/" << std::to_string(i) << ".txt" 
    //                   << " can't open" <<  std::endl;
    //         continue;
    //     }
    //     for (int j = 0; j < _segments[i].size(); ++j) {
    //         if (!_segments[i][j].hasPoint()) {
    //             continue;
    //         }
    //         file << _segments[i][j].getGroundPoint().d << " "
    //              << _segments[i][j].getGroundPoint().z << "\n";
    //     }
    //     file.close();
    // }
}
}