#include "ros/ros.h"
#include <ros/package.h>

#include <pcl/io/pcd_io.h>
#include <pcl/io/png_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>

#include <iostream>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>

#include "harp_msgs/perceptionData.h"

#include "perception_tools.h" 
#include "depth_smoother.h"

using namespace std;


harp::perception::DepthImageSmoother::DepthImageSmoother () {
    colmap.resize(w,1);
    rowmap.resize(h,1);
    prepareLookUpTable();
}


void harp::perception::DepthImageSmoother::prepareLookUpTable()
{
    float * pm1 = colmap.data();
    float * pm2 = rowmap.data();
    for(int i = 0; i < w; i++) {
        *pm1++ = (i-cx + 0.5) / fx;
    }
    for (int i = 0; i < h; i++) {
        *pm2++ = (i-cy + 0.5) / fy;
    }
    return;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr harp::perception::DepthImageSmoother::depthmapToPointcloud (
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in, 
    cv::Mat inpainted_depth_image) {

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_out (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::copyPointCloud(*cloud_in, *cloud_out);

  for(std::size_t y = 0; y < h; ++y){
    
    const unsigned int offset = y * w;
    const float dy = rowmap(y);

    for(std::size_t x = 0; x < w; ++x) {

      const float depth_value = inpainted_depth_image.at<double>(y,x);

      if(!std::isnan(depth_value) && !(std::abs(depth_value) < 0.0001)){
        const float rx = colmap(x) * depth_value;
        const float ry = dy * depth_value;
        cloud_out->points[offset+x].z=depth_value;
        cloud_out->points[offset+x].x=rx;
        cloud_out->points[offset+x].y=ry;
      }
    }
  }
  return cloud_out;
}

bool harp::perception::DepthImageSmoother::isPointValid(const pcl::PointXYZRGB &point) { 
  bool valid = true;
  if (std::isnan(point.x) ||
      std::isinf(point.x) ||
      std::isnan(point.y) ||
      std::isinf(point.y) ||
      std::isnan(point.z) ||
      std::isinf(point.z)) {
    valid = false;
  }
  return valid;
}

void harp::perception::DepthImageSmoother::inpaintDepthImage(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_in, const cv::Mat &mask, double max_range, cv::Mat &inpainted_depth_image) {

  const int width = cloud_in->width;
  const int height = cloud_in->height;

  if (height == 1)  {
    cerr << "Input PCD is not organized " << endl;
    return;
  }

  cv::Mat depth_image, depth_image_original;
  cv::Mat invalid_pixel_mask;
  depth_image.create(height, width, CV_8UC1);
  depth_image_original.create(height, width, CV_64FC1);
  invalid_pixel_mask.create(height, width, CV_8UC1);

  for (int y = 0; y < height; ++y) {
    auto y_ptr = depth_image.ptr<uchar>(y);
    auto original_y_ptr = depth_image_original.ptr<double>(y);
    auto mask_y_ptr = invalid_pixel_mask.ptr<uchar>(y);

    for (int x = 0; x < width; ++x) {
      const int pcl_index = y * width + x;
      const pcl::PointXYZRGB point = cloud_in->points[pcl_index];
      const double depth = point.z;
      original_y_ptr[x] = depth;

      if (isPointValid(point)) {
        mask_y_ptr[x] = 0;

        if (depth < max_range) {
          y_ptr[x] = static_cast<uchar>(std::min(depth, max_range) * 255 / max_range);
        }
      } else {
        y_ptr[x] = 0;
        mask_y_ptr[x] = 1;
      }
    }
  }

  const double kResizeScale = 0.1;

  cv::Mat resized_depth_image, resized_inpainted_mask;
  cv::resize(depth_image, resized_depth_image, cv::Size(), kResizeScale,
             kResizeScale);

  cv::Mat inpainted_mask = (mask > 0) & (invalid_pixel_mask > 0);
  cv::resize(inpainted_mask, resized_inpainted_mask, cv::Size(), kResizeScale,
             kResizeScale);

  static cv::Mat inpainted;
  cv::inpaint(resized_depth_image, resized_inpainted_mask, inpainted, 5,
              cv::INPAINT_NS);

  cv::resize(inpainted, inpainted, cv::Size(), 1 / kResizeScale,
             1 / kResizeScale);

  // Now copy over the inpainted pixels to the original depth image (and
  // cloud).
  cv::Mat depth_image_smoothed;
  depth_image_smoothed = depth_image_original.clone();

  for (int y = 0; y < height; ++y) {
    auto inpainted_y_ptr = inpainted.ptr<uchar>(y);
    auto smoothed_y_ptr = depth_image_smoothed.ptr<double>(y);
    auto mask_y_ptr = inpainted_mask.ptr<uchar>(y);

    for (int x = 0; x < width; ++x) {
      if (!mask_y_ptr[x]) {
        continue;
      }

      const double depth = static_cast<double>(inpainted_y_ptr[x]) * max_range / 255.0;
      smoothed_y_ptr[x] = depth;

    }
  }
  
  inpainted_depth_image = depth_image_smoothed;

}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr harp::perception::DepthImageSmoother::smoothPointCloud(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in,
    cv::Mat image_in) {

    // Inpaint required greyscale image
    cv::Mat bin_mask;
    cv::cvtColor(image_in, bin_mask, CV_BGR2GRAY);

    // Compute inpainted image
    cv::Mat inpainted_depth_image;
    const double max_range = 1.5; //Ignore points beyond this range for inpainting
    inpaintDepthImage(cloud_in, bin_mask, max_range, inpainted_depth_image);  

    // Map inpainted image back to depth cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr reconstruction_out;
    reconstruction_out = depthmapToPointcloud(cloud_in, inpainted_depth_image);

    return reconstruction_out;
}