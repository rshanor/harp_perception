#ifndef ROS_PERCEPTION_H
#define ROS_PERCEPTION_H

#include "ros/ros.h"
#include <pcl/io/pcd_io.h>
#include <string>
#include <tf/transform_listener.h>

namespace harp {
namespace perception {

    void publishCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr image, 
        ros::Publisher pub, 
        std::string frame);
    void publishCloud(
        pcl::PointCloud<pcl::PointXYZ>::Ptr image, 
        ros::Publisher pub, 
        std::string frame);
    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformCloud (
        tf::StampedTransform t,
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr input);

    Eigen::Affine3f readMatrix(
        std::string inputFile, 
        int startLine);


}
}

#endif
