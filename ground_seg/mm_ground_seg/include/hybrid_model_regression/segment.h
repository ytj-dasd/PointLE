#ifndef MM_GROUND_SEGMENTATION_SEGMENT_H_
#define MM_GROUND_SEGMENTATION_SEGMENT_H_

#include <list>
#include <map>

#include "common.hpp"
#include "bin.h"

namespace mm{
class Segment {

public:
    typedef std::pair<GroundPoint, GroundPoint> Line;
    typedef std::pair<double, double> LocalLine;

    Segment(const unsigned int& n_bins,
            const double& max_slope,
            const double& max_error,
            const double& long_threshold,
            const double& max_long_height,
            const double& max_start_height,
            const double& sensor_height,
            const double& bin_length);
    
    void fitSegmentLines();
    void originFitSegmentLines();

    double verticalDistanceToLine(const double& d, const double &z);

    bool getLines(std::list<Line>* lines);

    inline Bin& operator[](const size_t& index) {return _bins[index];}
    inline std::vector<Bin>::iterator begin() {return _bins.begin();}
    inline std::vector<Bin>::iterator end() {return _bins.end();}
    inline size_t size(){return _bins.size();}

private:
    LocalLine fitLocalLine(const std::list<GroundPoint>& points);

    Line localLineToLine(const LocalLine& local_line, const std::list<GroundPoint>& line_points);

    double getMeanError(const std::list<GroundPoint>& points, const LocalLine& line);

    double getMaxError(const std::list<GroundPoint>& points, const LocalLine& line);

    bool findLineBegin(double& cur_ground_height, std::vector<Bin>::iterator& it);
    
private:
    std::vector<Bin> _bins;
    std::list<Line> _lines;
    // Parameters. Description in GroundSegmentation.
    const double _sensor_height;
    const double _ground_thresh = 0.5;
    const double _max_slope;
    const double _max_error;
    const double _long_threshold;
    const double _max_long_height;
    const double _max_start_height;
    const double _bin_length;
};
}
#endif /* GROUND_SEGMENTATION_SEGMENT_H_ */
