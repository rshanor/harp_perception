
#include "ros/ros.h"

#include <iostream>
#include <string>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>

#include <opencv2/opencv.hpp>

#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include <tf/transform_listener.h>

#include "perception_tools.h"
#include "harp_msgs/perceptionData.h"
#include "harp_msgs/geometryFilterData.h"

#include <sensor_msgs/PointCloud2.h>

#include "depth_smoother.h"
#include "geometry_filter.h"

using namespace cv;
using namespace std;
using namespace pcl;
using namespace harp::perception;

harp::perception::GeometryFilter::GeometryFilter () {
    
    ros::param::get("geometry_filter_dx_min",dx_min);
    ros::param::get("geometry_filter_dy_min",dy_min);
    ros::param::get("geometry_filter_dz_min",dz_min);

    ros::param::get("geometry_filter_dx_max",dx_max);
    ros::param::get("geometry_filter_dy_max",dy_max);
    ros::param::get("geometry_filter_dz_max",dz_max);

    ros::param::get("bin_2_num_dict",bin_2_num_dict);
}

bool harp::perception::GeometryFilter::isValidPoint (float x, float y, float z, bin_frame bounds)
{

    if (x < bounds.x_min) return false;
    if (x > bounds.x_max) return false;
    if (y < bounds.y_min) return false;
    if (y > bounds.y_max) return false;
    if (z < bounds.z_min) return false;    
    if (z > bounds.z_max) return false;
    
    return true;
}   

bin_frame harp::perception::GeometryFilter::getMinMax (string current_bin)
{
    bin_frame bounds;
    
    int bin_of_interest = bin_2_num_dict[current_bin];

    int shelf_width = 3; // # of Bins wide
    int shelf_height = 4; // # of Bins tall
    int number_of_bins = shelf_width * shelf_height;

    float D = .43; // Meters -- Y Direction

    float H_tall = .2669;
    float H_short = .229;
    float W_wide = .304;
    float W_narrow = .276;

    float bin_width;
    float bin_height;
    int row_of_interest = bin_of_interest / shelf_width;
    int col_of_interest = bin_of_interest % shelf_width;

    switch(row_of_interest){
      case 0:
        bin_height = H_tall;
        break;
      case 1:
        bin_height = H_short;
        break;
      case 2:
        bin_height = H_short;
        break;
      case 3:
        bin_height = H_tall;
        break;
    }

    switch(col_of_interest){
      case 0:
        bin_width = W_narrow;
        break;
      case 1:
        bin_width = W_wide;
        break;
      case 2:
        bin_width = W_narrow;
        break;
    }


    bounds.z_min = 0 +  dz_min;
    bounds.z_max = bin_height - dz_max;

    bounds.x_min = 0 + dx_min;
    bounds.x_max = bin_width - dx_max;

    bounds.y_min = 0 + dy_min;
    bounds.y_max = D - dy_max;

    return bounds;
}

string harp::perception::GeometryFilter::getBin(float rowDist, float colDist)
{
    string bin;

    if ((rowDist > 1.55) and (rowDist < 1.75) and (colDist > .2) and (colDist < .4)) bin = "bin_A";
    if ((rowDist > 1.55) and (rowDist < 1.75) and (colDist > -0.1) and (colDist < 0.1)) bin = "bin_B";
    if ((rowDist > 1.55) and (rowDist < 1.75) and (colDist > -0.4) and (colDist < -0.15)) bin = "bin_C";

    if ((rowDist > 1.25) and (rowDist < 1.45) and (colDist > .2) and (colDist < .4)) bin = "bin_D";
    if ((rowDist > 1.25) and (rowDist < 1.45) and (colDist > -0.1) and (colDist < 0.1)) bin = "bin_E";
    if ((rowDist > 1.25) and (rowDist < 1.45) and (colDist > -0.4) and (colDist < -0.15)) bin = "bin_F";

    if ((rowDist > 1.0) and (rowDist < 1.25) and (colDist > .2) and (colDist < .4)) bin = "bin_G";
    if ((rowDist > 1.0) and (rowDist < 1.25) and (colDist > -0.1) and (colDist < 0.1)) bin = "bin_H";
    if ((rowDist > 1.0) and (rowDist < 1.25) and (colDist > -0.4) and (colDist < -0.15)) bin = "bin_I";

    if ((rowDist > 0.7) and (rowDist < 1.0) and (colDist > .2) and (colDist < .4)) bin = "bin_J";
    if ((rowDist > 0.7) and (rowDist < 1.0) and (colDist > -0.1) and (colDist < 0.1)) bin = "bin_K";
    if ((rowDist > 0.7) and (rowDist < 1.0) and (colDist > -0.4) and (colDist < -0.15)) bin = "bin_L";

    return bin;
}

Mat harp::perception::GeometryFilter::filterImage(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, 
    Mat image, 
    string bin) {

    Mat output_image;
    image.copyTo(output_image);

    bin_frame bounds;
    bounds = getMinMax(bin);

    float x, y, z;
    int index_row, index_col;

    for (int i = 0; i < cloud->points.size(); i++) {
        // Transform from cloud index to image index
        index_row = (int) i / image.cols;
        index_col = (int) i % image.cols;
        // Get X,Y,Z coords of point
        x = cloud->points[i].x;
        y = cloud->points[i].y;
        z = cloud->points[i].z;
        // Check if point is valid
        if (!isValidPoint(x,y,z,bounds)) {
            output_image.at<cv::Vec3b>(index_row,index_col)[0] = 0;
            output_image.at<cv::Vec3b>(index_row,index_col)[1] = 0;
            output_image.at<cv::Vec3b>(index_row,index_col)[2] = 0;
        }
    }

    return output_image;
}