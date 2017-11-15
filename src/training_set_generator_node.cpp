#include <iostream>
#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <Eigen/Core>
#include <math.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "srrg_path_map/path_map.h"
#include "srrg_path_map/path_map_utils.h"
#include "srrg_path_map/distance_map_path_search.h"
#include "srrg_system_utils/system_utils.h"
#include "tf/tf.h"
#include "tf/transform_listener.h"
#include "tf/transform_datatypes.h"
#include <sensor_msgs/CameraInfo.h>
#include <semantic_map_benchmarking/LogicalCameraImage.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/passthrough.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <gazebo_msgs/SetModelState.h>

using namespace std;
using namespace srrg_core;
using namespace message_filters;
using namespace semantic_map_benchmarking;

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

class TrainingSetGenerator {
public:
  TrainingSetGenerator ():
    _logical_image_sub (_nh,"/gazebo/logical_camera_image",1),
    _depth_cloud_sub (_nh,"/camera/depth/points",1),
    _rgb_image_sub (_nh,"/camera/rgb/image_raw", 1),
    _synchronizer (FilterSyncPolicy (10),_logical_image_sub,_depth_cloud_sub,_rgb_image_sub){
    
    _occ_threshold = 60;
    _free_threshold = 240;
    _map_sub = _nh.subscribe ("/map", 1000, &TrainingSetGenerator::mapCallback, this);

    _got_info = false;
    _camera_info_sub = _nh.subscribe ("/camera/depth/camera_info",
				      1000,
				      &TrainingSetGenerator::cameraInfoCallback,
				      this);

    _synchronizer.registerCallback (boost::bind (&TrainingSetGenerator::filterCallback, this, _1, _2, _3));
    _set_model_state_client = _nh.serviceClient<gazebo_msgs::SetModelState> ("set_model_state");
  }

  void mapCallback(const nav_msgs::OccupancyGrid::ConstPtr& map_msg){

    _resolution = map_msg->info.resolution;
    _origin = Eigen::Vector2f (map_msg->info.origin.position.x,
			       map_msg->info.origin.position.y);
    int width = map_msg->info.width;
    int height = map_msg->info.height;
    _size = Eigen::Vector2i (width,height);
    
    cv::Mat map_image = cv::Mat(height, width, CV_8U);
    ROS_INFO("Occupancy grid received.");
    for (int i = 0, i_rev = height - 1; i < height; i++, i_rev--)
      for (int j = 0; j < width; j++)
	switch (map_msg->data[i_rev*width + j]) {
	default:
	case -1:
	  map_image.data[i*width + j] = 150;
	  break;
	case 0:
	  map_image.data[i*width + j] = 255;
	  break;
	case 100:
	  map_image.data[i*width + j] = 0;
	  break;
	}

    IntImage indices_image;
    grayMap2indices(indices_image, map_image, _occ_threshold, _free_threshold);
    float safety_region = 1.0f;
    indices2distances(_distance_image, indices_image, _resolution, safety_region);
    
    _overlay_resolution = 1.0f;
    float ratio = _resolution/_overlay_resolution;
    
    _overlay_origin = Eigen::Vector2f (map_msg->info.origin.position.x,
				       map_msg->info.origin.position.y);
    
    _overlay_size = Eigen::Vector2i (floor(width*ratio),
				     floor(height*ratio));
    _got_map = true;
    _map_sub.shutdown();

    cv::Mat output;
    _distance_image*=(1.f/safety_region);
    _distance_image.convertTo(output,CV_8UC1,255);
    cv::imwrite("distance_map.png",output);
  }

  
  void cameraInfoCallback(const sensor_msgs::CameraInfo::ConstPtr& camera_info_msg){
    sensor_msgs::CameraInfo camerainfo;
    camerainfo.K = camera_info_msg->K;

    ROS_INFO("Got camera info!");
    _K(0,0) = camerainfo.K.c_array()[0];
    _K(0,1) = camerainfo.K.c_array()[1];
    _K(0,2) = camerainfo.K.c_array()[2];
    _K(1,0) = camerainfo.K.c_array()[3];
    _K(1,1) = camerainfo.K.c_array()[4];
    _K(1,2) = camerainfo.K.c_array()[5];
    _K(2,0) = camerainfo.K.c_array()[6];
    _K(2,1) = camerainfo.K.c_array()[7];
    _K(2,2) = camerainfo.K.c_array()[8];

    cerr << _K << endl;

    _got_info = true;
    _camera_info_sub.shutdown();
  }

  void filterCallback(const semantic_map_benchmarking::LogicalCameraImage::ConstPtr& logical_image_msg,
		      const PointCloud::ConstPtr& scene_cloud_msg,
		      const sensor_msgs::Image::ConstPtr& rgb_image_msg){
    if(_got_info){
      
      cv_bridge::CvImageConstPtr rgb_cv_ptr;
      try{
	rgb_cv_ptr = cv_bridge::toCvShare(rgb_image_msg);
      } catch (cv_bridge::Exception& e) {
	ROS_ERROR("cv_bridge exception: %s", e.what());
	return;
      }

      _rgb_image = rgb_cv_ptr->image.clone();
      
      tf::StampedTransform depth_camera_pose;
      try {
	_listener.waitForTransform("map",
				   "camera_depth_optical_frame",
				   ros::Time(0),
				   ros::Duration(3));
	_listener.lookupTransform("map",
				  "camera_depth_optical_frame",
				  ros::Time(0),
				  depth_camera_pose);
      }
      catch(tf::TransformException ex) {
	ROS_ERROR("%s", ex.what());
      }

      Eigen::Isometry3f depth_camera_transform = tfTransform2eigen(depth_camera_pose);
      pcl::transformPointCloud (*scene_cloud_msg, _map_cloud, depth_camera_transform);
      _map_cloud.header.frame_id = "/map";
      _map_cloud.width  = scene_cloud_msg->width;
      _map_cloud.height = scene_cloud_msg->height;
      _map_cloud.is_dense = false;

      _logical_image = *logical_image_msg;
    }
    
  }


  void generate(){
    int i=0;
    int j=0;

    while(_nh.ok()){

      if(isFree(i,j)){
	gazebo_msgs::SetModelState srv;

	if(_set_model_state_client.call(srv)){
	  ROS_INFO("response");
	} else {
	  ROS_ERROR("Failed to call service set_model_state");
	  return;
	}
	
      }
      
    }
  }
  
private:
  ros::NodeHandle _nh;

  ros::Subscriber _map_sub;
  int _occ_threshold;
  int _free_threshold;
  float _resolution;
  Eigen::Vector2f _origin;
  Eigen::Vector2i _size;
  float _overlay_resolution;
  Eigen::Vector2f _overlay_origin;
  Eigen::Vector2i _overlay_size;
  FloatImage _distance_image;
  bool _got_map;

  ros::Subscriber _camera_info_sub;
  Eigen::Matrix3f _K;
  bool _got_info;

  tf::TransformListener _listener;
  message_filters::Subscriber<LogicalCameraImage> _logical_image_sub;
  message_filters::Subscriber<PointCloud> _depth_cloud_sub;
  message_filters::Subscriber<sensor_msgs::Image> _rgb_image_sub;
  typedef sync_policies::ApproximateTime<LogicalCameraImage,PointCloud,sensor_msgs::Image> FilterSyncPolicy;
  message_filters::Synchronizer<FilterSyncPolicy> _synchronizer;

  cv::Mat _rgb_image;
  PointCloud _map_cloud;
  LogicalCameraImage _logical_image;
  ros::ServiceClient _set_model_state_client;
  
  Eigen::Isometry3f tfTransform2eigen(const tf::Transform& p){
    Eigen::Isometry3f iso;
    iso.translation().x()=p.getOrigin().x();
    iso.translation().y()=p.getOrigin().y();
    iso.translation().z()=p.getOrigin().z();
    Eigen::Quaternionf q;
    tf::Quaternion tq = p.getRotation();
    q.x()= tq.x();
    q.y()= tq.y();
    q.z()= tq.z();
    q.w()= tq.w();
    iso.linear()=q.toRotationMatrix();
    return iso;
  }

  bool isFree(int i, int j){
    float x = _overlay_origin.x() + i*_overlay_resolution;
    float y = _overlay_origin.y() + j*_overlay_resolution;

    int r = (x - _origin.x())/_resolution;
    int c = (y - _origin.y())/_resolution;

    return (_distance_image.at<float>(r,c) > 0.8);
  }

  
};

int main(int argc, char** argv){
  ros::init(argc,argv,"training_set_generator");
  TrainingSetGenerator generator;
  ros::spin();
  return 0;
}
