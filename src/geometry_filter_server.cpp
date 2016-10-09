/**
 * @file geometry_filter_server.cpp
 * @brief Filter out shelf based on depth
 * @author Rick Shanor
 * Carnegie Mellon University, 2016
 */

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

#include "harp_msgs/perceptionData.h"
#include "harp_msgs/geometryFilterData.h"

#include <sensor_msgs/PointCloud2.h>

#include "depth_smoother.h"
#include "geometry_filter.h"
#include "perception_tools.h"

using namespace cv;
using namespace std;
using namespace pcl;
using namespace harp::perception;

class GeometeryFilterServer
{
    ros::NodeHandle nh;
    image_transport::ImageTransport it;
    ros::ServiceServer service;
    tf::TransformListener listener;

public:
    GeometeryFilterServer(): it(nh) {
        service = nh.advertiseService("geometry_filter_server", &GeometeryFilterServer::runGeometryFilter, this);
    }

private:
    bool runGeometryFilter (harp_msgs::geometryFilterData::Request&  req,
                    harp_msgs::geometryFilterData::Response& res);

};

bool GeometeryFilterServer::runGeometryFilter(harp_msgs::geometryFilterData::Request&  req,
                                       harp_msgs::geometryFilterData::Response& res)
{

    // Load point cloud from file
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    string cloud_path = req.cloud_path;
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (cloud_path, *cloud) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
        return (-1);
    }

    // Load premasked image
    string image_path = req.image_path;
    Mat image = imread(image_path, CV_LOAD_IMAGE_COLOR);   // Read the file

    // Load TF data
    Eigen::Affine3f kinect_2_world;
    Eigen::Affine3f shelf_2_world;
    string tf_path = req.tf_path;
    int startLine = 2;
    kinect_2_world = readMatrix(tf_path,startLine);
    startLine = 12;
    shelf_2_world = readMatrix(tf_path,startLine);

    // Transfrom cloud 
    Eigen::Matrix4f M = kinect_2_world.inverse().matrix() * shelf_2_world.matrix();
    pcl::transformPointCloud (*cloud, *cloud, M);

    // Get target bin from kinect position
    // Hack, data should have been saved with bin info
    harp::perception::GeometryFilter gf;
    float rowDist = kinect_2_world.matrix()(2,3);
    float colDist = kinect_2_world.matrix()(1,3);
    string bin = gf.getBin(rowDist, colDist);

    // Smooth point cloud based on depth data
    harp::perception::DepthImageSmoother dis;
    cloud = dis.smoothPointCloud(cloud,image); 

    // Transfrom cloud to correct frame
    M = M.inverse();
    pcl::transformPointCloud (*cloud, *cloud, M);
    tf::StampedTransform shelf_2_bin;
    ros::Time now = ros::Time::now();
    listener.waitForTransform(bin, "/shelf",
                          now, ros::Duration(10.0));
    listener.lookupTransform(bin, "/shelf",
                         now, shelf_2_bin);
    cloud = transformCloud(shelf_2_bin, cloud);    

    // Run masking algorithm
    Mat output_image = gf.filterImage(cloud,image,bin);

    // Convert result to image message
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", output_image).toImageMsg();
    res.img_out = *msg;

    return true;
}


int main(int argc, char **argv) {

  ros::init(argc, argv, "geometry_filter_server");
  
  GeometeryFilterServer gfs;

  ros::Rate loop_rate(1);
  while (ros::ok())
  {
    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;  
}


