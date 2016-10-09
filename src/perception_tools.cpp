#include "ros/ros.h"
#include "perception_tools.h" 
#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <string>
#include <fstream>
#include <sstream>
#include <pcl/common/transforms.h>

using namespace std;

// Publish XYZRGB pointcloud
void harp::perception::publishCloud(
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr image, 
  ros::Publisher pub, 
  std::string frame) {
  sensor_msgs::PointCloud2 bin_msg;
  pcl::toROSMsg(*image, bin_msg);
  bin_msg.header.stamp = ros::Time::now();
  bin_msg.header.frame_id = frame;
  bin_msg.height = 1;
  bin_msg.width = image->points.size();
  pub.publish (bin_msg);
  return;  
}

// Publish XYZ pointcloud
void harp::perception::publishCloud(
  pcl::PointCloud<pcl::PointXYZ>::Ptr image, 
  ros::Publisher pub, 
  std::string frame) {
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr image_new (new pcl::PointCloud<pcl::PointXYZRGB> ());
	pcl::copyPointCloud(*image, *image_new);
	publishCloud(image_new, pub, frame);
	return;
}

// Transform scene (RGB) to new TF
pcl::PointCloud<pcl::PointXYZRGB>::Ptr harp::perception::transformCloud (
  tf::StampedTransform t,
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr input) {

  float x = t.getOrigin().x();
  float y = t.getOrigin().y(); 
  float z = t.getOrigin().z();

  tf::Quaternion q;
  q = t.getRotation();

  double roll, pitch, yaw;
  tf::Matrix3x3(q).getRPY(roll, pitch, yaw);

  Eigen::Affine3f A;
  pcl::getTransformation (x,y,z,roll,pitch,yaw,A);

  Eigen::Matrix4f M;
  M = A.matrix();

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr new_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::transformPointCloud (*input, *new_cloud, M);
  return new_cloud;
}


Eigen::Affine3f harp::perception::readMatrix(string inputFile, int startLine)
{

    ifstream myfile (inputFile);
    Eigen::Affine3f mat;
    Eigen::Matrix4f temp;
    int ct = 1;
    int linesRead = 0;
    string line;
    if (myfile.is_open())
    {
        while (! myfile.eof() )
        {
            getline (myfile,line);
            if (ct >= startLine && ct < startLine+4)
            {
                int charRead = 0;
                istringstream iss(line);
                while(iss)
                {
                    string subs;
                    iss >> subs;
                    if (subs.length() > 0)
                    {
                    float m1 = std::stof (subs);
                    temp(linesRead,charRead) = m1;
                    charRead ++;

                    }
                }
            linesRead ++;
            }
        ct ++;
        
        }
    myfile.close();

    }
    mat.matrix()=temp;
    return mat;    
}