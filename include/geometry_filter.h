#ifndef __geometry__filter__
#define __deometry__filter__

#include <pcl/io/pcd_io.h>
#include <opencv2/opencv.hpp>
#include <string>

using namespace std;

namespace harp {
namespace perception {

struct bin_frame{
    float x_min;
    float x_max;
    float y_min;
    float y_max;
    float z_min;
    float z_max; 
};

class GeometryFilter {

public:

    GeometryFilter();

    cv::Mat filterImage (
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in,
        cv::Mat image_in,
        string bin);

    string getBin(float rowDist, float colDist);


private:

    float dx_min, dy_min, dz_min;
    float dx_max, dy_max, dz_max;
    std::map<std::string, int> bin_2_num_dict;

    bool isValidPoint (float x, float y, float z, bin_frame bounds);
    bin_frame getMinMax (string current_bin);


}; // End of class definition

} // End of perception namespace
} // End of harp namespace

#endif