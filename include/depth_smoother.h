#ifndef __depth__smoother__
#define __depth__smoother__

#include <pcl/io/pcd_io.h>
#include <opencv2/opencv.hpp>
#include <pcl_ros/point_cloud.h>

namespace harp {
namespace perception {

class DepthImageSmoother {

public:

    DepthImageSmoother();

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr smoothPointCloud(
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in,
        cv::Mat imageIn);

private:

    // Camera and image parameters
    const float cx = 945.58752751085558;
    const float cy = 520.06994012356529;
    const float fx = 1066.01203606379;
    const float fy = 1068.87083999116;    

    const int w = 1920;
    const int h = 1080;    

    Eigen::MatrixXf colmap;
    Eigen::MatrixXf rowmap;

    // Create lookup table to map depth image to point cloud    
    void prepareLookUpTable();

    // Convert from depth map to point cloud 
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr depthmapToPointcloud (
        const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in, 
        cv::Mat inpainted_depth_image);

    // Check if point is valid
    bool isPointValid(const pcl::PointXYZRGB &point);

    void inpaintDepthImage(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_in, const cv::Mat &mask, double max_range, cv::Mat &inpainted_depth_image); 

}; // End of class definition

} // End of perception namespace
} // End of harp namespace

#endif