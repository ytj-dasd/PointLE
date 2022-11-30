#include "hybrid_model_regression/segment.h"
namespace mm{
Segment::Segment(const unsigned int& n_bins,
                 const double& max_slope,
                 const double& _max_error,
                 const double& long_threshold,
                 const double& max_long_height,
                 const double& max_start_height,
                 const double& sensor_height,
                 const double& bin_length) :
                 _bins(n_bins),
                 _max_slope(max_slope),
                 _max_error(_max_error),
                 _long_threshold(long_threshold),
                 _max_long_height(max_long_height),
                 _max_start_height(max_start_height),
                 _sensor_height(sensor_height),
                 _bin_length(bin_length){}


bool Segment::findLineBegin(double& cur_ground_height, std::vector<Bin>::iterator& it) {
    bool has_finded = true;
    while (!it->hasPoint() ||
           std::fabs(it->getGroundPoint().z - cur_ground_height) > _max_start_height) {
        ++it;
        // Stop if we reached last point.
        if (it == _bins.end()) {
            has_finded = false;
            break;
        }
    }
    return has_finded;
}

void Segment::fitSegmentLines() {
    double cur_ground_height = -_sensor_height;
    bool is_long_line = false;
    std::list<GroundPoint> current_line_points;
    LocalLine cur_line = std::make_pair(0,0);
    for (auto line_iter = _bins.begin(); line_iter != _bins.end(); ++line_iter) {
        if (!line_iter->hasPoint()) {
            continue;
        }
        if (current_line_points.size() == 0){
            if(!findLineBegin(cur_ground_height, line_iter)) {
               break;
            }
            current_line_points.push_back(line_iter->getGroundPoint());
            cur_ground_height = line_iter->getGroundPoint().z;
        }else if (current_line_points.size() == 1) {
            GroundPoint cur_point = line_iter->getGroundPoint();
            if (std::fabs(cur_point.z - cur_ground_height) < _max_start_height) {
                current_line_points.push_back(cur_point);
                cur_ground_height = cur_point.z;
            }else {
                current_line_points.clear();
                --line_iter;
            }
        }else if (current_line_points.size() > 1) {
            GroundPoint cur_point = line_iter->getGroundPoint();
            is_long_line = 
                cur_point.d - current_line_points.back().d > _long_threshold;
            // Get expected z value to possibly reject far away points.
            double expected_z = cur_ground_height;
            if (is_long_line) {
                expected_z = cur_line.first * cur_point.d + cur_line.second;
            }
            current_line_points.push_back(cur_point);
            cur_line = fitLocalLine(current_line_points);
            const double error = getMaxError(current_line_points, cur_line);
            // Check if not a good line.
            if (error > _max_error ||
                std::fabs(cur_line.first) > _max_slope ||
                std::fabs(expected_z - cur_point.z) > _max_long_height) {
                // Add line until previous point as ground.
                current_line_points.pop_back();
                // Don't let lines with 2 base points through.
                const LocalLine new_line = fitLocalLine(current_line_points);
                _lines.push_back(localLineToLine(new_line, current_line_points));
                cur_ground_height = new_line.first * current_line_points.back().d + new_line.second;
                // Start new line.
                current_line_points.clear();
                --line_iter;
            }
        }
    }
    // Add last line.
    if (current_line_points.size() > 2) {
        const LocalLine new_line = fitLocalLine(current_line_points);
        _lines.push_back(localLineToLine(new_line, current_line_points));
    }
}

void Segment::originFitSegmentLines() {
    // Find first point.
    double cur_ground_height = -_sensor_height;
    auto line_start = _bins.begin();
    while (!line_start->hasPoint() || 
            std::fabs(line_start->getGroundPoint().z - cur_ground_height) > _max_start_height) {
        ++line_start;
        // Stop if we reached last point.
        if (line_start == _bins.end()) return;
    }
    // Fill lines.
    bool is_long_line = false;
    cur_ground_height = line_start->getGroundPoint().z;
    std::list<GroundPoint> current_line_points(1, line_start->getGroundPoint());
    LocalLine cur_line = std::make_pair(0,0);
    for (auto line_iter = line_start+1; line_iter != _bins.end(); ++line_iter) {
        if (!line_iter->hasPoint()) {
            continue;
        }
        GroundPoint cur_point = line_iter->getGroundPoint();
        if (cur_point.d - current_line_points.back().d > _long_threshold) is_long_line = true;
        if (current_line_points.size() >= 2) {
            // Get expected z value to possibly reject far away points.
            double expected_z = std::numeric_limits<double>::max();
            if (is_long_line && current_line_points.size() > 2) {
                expected_z = cur_line.first * cur_point.d + cur_line.second;
            }
            current_line_points.push_back(cur_point);
            cur_line = fitLocalLine(current_line_points);
            const double error = getMaxError(current_line_points, cur_line);
            // Check if not a good line.
            if (error > _max_error ||
                std::fabs(cur_line.first) > _max_slope ||
                is_long_line && std::fabs(expected_z - cur_point.z) > _max_long_height) {

                // Add line until previous point as ground.
                current_line_points.pop_back();
                // Don't let lines with 2 base points through.
                if (current_line_points.size() >= 3) {
                    const LocalLine new_line = fitLocalLine(current_line_points);
                    _lines.push_back(localLineToLine(new_line, current_line_points));
                    cur_ground_height = new_line.first * current_line_points.back().d + new_line.second;
                }
                // Start new line.
                is_long_line = false;
                current_line_points.clear();
                --line_iter;
            }
        }else {
            // Not enough points.
            if (cur_point.d - current_line_points.back().d < _long_threshold &&
                std::fabs(current_line_points.back().z - cur_ground_height) < _max_start_height) {
                // Add point if valid.
                current_line_points.push_back(cur_point);
            } else {
                // Start new line.
                current_line_points.clear();
                current_line_points.push_back(cur_point);
            }
        }
    }
    // Add last line.
    if (current_line_points.size() > 2) {
        const LocalLine new_line = fitLocalLine(current_line_points);
        _lines.push_back(localLineToLine(new_line, current_line_points));
    }
}

double Segment::verticalDistanceToLine(const double &d, const double &z) {
    static const double kMargin = _bin_length;
    double distance = -1;
    for (auto it = _lines.begin(); it != _lines.end(); ++it) {
        if (it->first.d - kMargin < d && it->second.d + kMargin > d) {
            const double delta_z = it->second.z - it->first.z;
            const double delta_d = it->second.d - it->first.d;
            const double expected_z = (d - it->first.d) / delta_d * delta_z + it->first.z;
            distance = std::fabs(z - expected_z);
        }
    }
    return distance;
}

bool Segment::getLines(std::list<Line> *lines) {
    if (_lines.empty()) {
       return false;
    }else {
        *lines = _lines;
        return true;
    }
}

double Segment::getMeanError(const std::list<GroundPoint> &points, const LocalLine &line) {
    double error_sum = 0;
    for (auto it = points.begin(); it != points.end(); ++it) {
        const double residual = (line.first * it->d + line.second) - it->z;
        error_sum += residual * residual;
    }
    return error_sum / points.size();
}

double Segment::getMaxError(const std::list<GroundPoint> &points, const LocalLine &line) {
    double _max_error = 0;
    for (auto it = points.begin(); it != points.end(); ++it) {
        const double residual = (line.first * it->d + line.second) - it->z;
        const double error = residual * residual;
        if (error > _max_error) {
            _max_error = error;
        }
    }
    return _max_error;
}

Segment::LocalLine Segment::fitLocalLine(const std::list<GroundPoint> &points) {
    const unsigned int n_points = points.size();
    Eigen::MatrixXd X(n_points, 2);
    Eigen::VectorXd Y(n_points);
    unsigned int counter = 0;
    for (auto iter = points.begin(); iter != points.end(); ++iter) {
        X(counter, 0) = iter->d;
        X(counter, 1) = 1;
        Y(counter) = iter->z;
        ++counter;
    }
    Eigen::VectorXd result = X.colPivHouseholderQr().solve(Y);
    LocalLine line_result;
    line_result.first = result(0);
    line_result.second = result(1);
    return line_result;
}

Segment::Line Segment::localLineToLine(const LocalLine& local_line,
                                       const std::list<GroundPoint>& line_points) {
    Line line;
    const double first_d = line_points.front().d;
    const double second_d = line_points.back().d;
    const double first_z = local_line.first * first_d + local_line.second;
    const double second_z = local_line.first * second_d + local_line.second;
    line.first.z = first_z;
    line.first.d = first_d;
    line.second.z = second_z;
    line.second.d = second_d;
    return line;
}
}