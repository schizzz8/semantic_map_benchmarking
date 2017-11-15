#include <iostream>
#include <sstream>
#include <fstream>
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
#include <gazebo_msgs/SetModelState.h>
#include <gazebo_msgs/ModelState.h>

using namespace std;
using namespace srrg_core;
using namespace semantic_map_benchmarking;

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;


constexpr unsigned int str2int(const char* str, int h = 0){
  return !str[h] ? 5381 : (str2int(str, h+1) * 33) ^ str[h];
}

int object2id(std::string object){
  switch (str2int(object.c_str())){
  case str2int("table"):
    return 1;
  case str2int("chair"):
    return 2;
  case str2int("bookcase"):
    return 3;
  case str2int("couch"):
    return 4;
  case str2int("cabinet"):
    return 5;
  case str2int("plant"):
    return 6;
  default:
    return 0;
  }

}

class TrainingSetGenerator {
  
public:
  TrainingSetGenerator (){

    _got_map = false;
    _occ_threshold = 60;
    _free_threshold = 240;
    _map_sub = _nh.subscribe ("/map", 1000, &TrainingSetGenerator::mapCallback, this);

    _got_info = false;
    _camera_info_sub = _nh.subscribe ("/camera/rgb/camera_info",
				      1000,
				      &TrainingSetGenerator::cameraInfoCallback,
				      this);

    _set_model_state_client = _nh.serviceClient<gazebo_msgs::SetModelState> ("/gazebo/set_model_state");

    _seq = 1;
    ROS_INFO("Starting training set generator node!");
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
    grayMap2indices (indices_image, map_image, _occ_threshold, _free_threshold);
    float safety_region = 1.0f;
    indices2distances (_distance_image, indices_image, _resolution, safety_region);
    ROS_INFO ("Distance map computed.");
    
    _overlay_resolution = 5.0f;
    float ratio = _resolution/_overlay_resolution;
    
    _overlay_origin = Eigen::Vector2f (map_msg->info.origin.position.x,
				       map_msg->info.origin.position.y);
    
    _overlay_size = Eigen::Vector2i (floor(width*ratio),
				     floor(height*ratio));
    ROS_INFO ("Defining overlay grid with following parameters:");
    cerr << "\t>> resolution: " << _overlay_resolution << endl;
    cerr << "\t>> size: " << _overlay_size.transpose() << endl;
    cerr << "\t>> origin: " << _overlay_origin.transpose() << endl;
    
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

  void generate(){
    bool stop=false;
    ros::Rate loop_rate(1);
    
    while(_nh.ok() && !stop){

      if(_got_map && _got_info){

	for(int i=0; i<_overlay_size.x(); i++)
	  for(int j=0; j<_overlay_size.y(); j++){
	    
	    if(isFree(i,j)){

	      //set robot pose
	      cerr << "Setting robot pose to: ("
		   << _overlay_origin.x() + i*_overlay_resolution << ","
		   << _overlay_origin.y() + j*_overlay_resolution << ")" << endl;
	      
	      gazebo_msgs::SetModelState set_model_state;

	      gazebo_msgs::ModelState model_state;
	      model_state.model_name = "robot";
	      model_state.reference_frame = "map";

	      geometry_msgs::Pose pose;
	      pose.position.x = _overlay_origin.x() + i*_overlay_resolution;
	      pose.position.y = _overlay_origin.y() + j*_overlay_resolution;
	      pose.position.z = 0;
	      model_state.pose = pose;

	      set_model_state.request.model_state = model_state;
	      
	      if(_set_model_state_client.call(set_model_state)){
		ROS_INFO("DONE!!!");
	      } else {
		ROS_ERROR("Failed to call service set_model_state: %s",set_model_state.response.status_message);
		return;
	      }

	      //receive messages
	      cv::Mat rgb_image;
	      PointCloud point_cloud;
	      LogicalCameraImage logical_image;

	      cerr << "rgb: " << receiveRgbImageMsg("/camera/rgb/image_raw",1,rgb_image) << endl;
	      cerr << "cloud: " << receivePointCloudMsg("/camera/depth/points",1,point_cloud) << endl;
	      cerr << "logical: " << receiveLogicalImageMsg("/gazebo/logical_camera_image",1,logical_image) << endl;
	      
	      if(receiveRgbImageMsg("/camera/rgb/image_raw",1,rgb_image) &&
		 receivePointCloudMsg("/camera/depth/points",1,point_cloud) &&
		 receiveLogicalImageMsg("/gazebo/logical_camera_image",1,logical_image)){

		if(logical_image.models.size() > 0){

		  //save image
		  stringstream ss;
		  ss << std::setw(6) << std::setfill('0') << _seq;
		  string filename = ss.str();
		  cv::imwrite(filename+".png",rgb_image);
		  cerr << "Saving image: " << filename+".png" << endl;

		  //open file to store detected objects
		  ofstream file;
		  file.open(filename+".txt");
            
		  tf::StampedTransform logical_camera_pose;
		  tf::poseMsgToTF(logical_image.pose,logical_camera_pose);
		
		  for (int idx=0; idx<logical_image.models.size(); ++idx){

		    //compute model bounding box
		    tf::Transform model_pose;
		    tf::poseMsgToTF(logical_image.models.at(idx).pose,model_pose);
		    Eigen::Isometry3f model_transform = tfTransform2eigen(logical_camera_pose)*tfTransform2eigen(model_pose);
		    pcl::PointXYZ min_pt,max_pt;
		    modelBoundingBox(logical_image.models.at(idx).min,
				     logical_image.models.at(idx).max,
				     model_transform,
				     min_pt,
				     max_pt);

		    //extract points that fall in the model bounding box
		    PointCloud::Ptr cloud_filtered_xyz (new PointCloud ());
		    filterPointCloud(point_cloud.makeShared(),
				     min_pt,
				     max_pt,
				     cloud_filtered_xyz);

		    //compute model bounding box in the rgb image
		    if(!cloud_filtered_xyz->points.empty()){
		      string object_type = logical_image.models.at(idx).type;
		      string object = object_type.substr(0,object_type.find_first_of("_"));

		      cv::Point2i p_min(10000,10000);
		      cv::Point2i p_max(-10000,-10000);
	  
		      for(int jdx=0; jdx<cloud_filtered_xyz->points.size(); ++jdx){
			Eigen::Vector3f camera_point = _depth_camera_transform.inverse()*
			  Eigen::Vector3f(cloud_filtered_xyz->points[jdx].x,
					  cloud_filtered_xyz->points[jdx].y,
					  cloud_filtered_xyz->points[jdx].z);
			Eigen::Vector3f image_point = _K*camera_point;

			const float& z=image_point.z();
			image_point.head<2>()/=z;
			int r = image_point.x();
			int c = image_point.y();

			if(r < p_min.x)
			  p_min.x = r;
			if(r > p_max.x)
			  p_max.x = r;

			if(c < p_min.y)
			  p_min.y = c;
			if(c > p_max.y)
			  p_max.y = c;

		      }//for cloud_filtered points
		    
		      file << object2id(object) << " ";
		      file << p_min.x << " " << p_min.y << " ";
		      file << p_max.x-p_min.x << " " << p_max.y-p_min.y << endl;
		    }// if cloud_filtered not empty
		  }//for models
		  file.close();
		  _seq++;
		}//if models not empty
	      }//if receive messages
	    }//if isfree
	    ros::spinOnce();
	    loop_rate.sleep();
	  }// for i,j
	stop=true;
      }// if got_map and got_info
      ros::spinOnce();
      loop_rate.sleep();
    }//while ros ok and not stop
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
  Eigen::Isometry3f _depth_camera_transform;

  cv::Mat _rgb_image;
  PointCloud _map_cloud;
  LogicalCameraImage _logical_image;
  ros::ServiceClient _set_model_state_client;

  int _seq;
  
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

    cerr << "Distance map at (" << i << "," << j << "): " << _distance_image.at<float>(r,c) << endl;

    return (_distance_image.at<float>(r,c) > 0.3);
  }

  bool receiveRgbImageMsg(const std::string& topic, float duration, cv::Mat& rgb_image){
    boost::shared_ptr<sensor_msgs::Image const> rgb_image_msg_ptr;
    rgb_image_msg_ptr = ros::topic::waitForMessage<sensor_msgs::Image> (topic, ros::Duration (duration));
    if(rgb_image_msg_ptr == NULL){
      ROS_ERROR ("No RGB image message received!!!");
      return false;
    }else{
      cv_bridge::CvImageConstPtr rgb_cv_ptr;
      try{
	rgb_cv_ptr = cv_bridge::toCvShare(rgb_image_msg_ptr);
      } catch (cv_bridge::Exception& e) {
	ROS_ERROR ("cv_bridge exception: %s", e.what());
	return false;
      }
      rgb_image = rgb_cv_ptr->image.clone();
      return true;
    }
  }

  bool receivePointCloudMsg(const std::string& topic, float duration, PointCloud& point_cloud){
    boost::shared_ptr<PointCloud const> point_cloud_msg_ptr;
    point_cloud_msg_ptr = ros::topic::waitForMessage<PointCloud> (topic, ros::Duration (duration));
    if(point_cloud_msg_ptr == NULL){
      ROS_ERROR ("No point cloud message received!!!");
      return false;
    }else{
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
      }catch(tf::TransformException ex) {
	ROS_ERROR("%s", ex.what());
	return false;
      }
      _depth_camera_transform = tfTransform2eigen(depth_camera_pose);
      pcl::transformPointCloud (*point_cloud_msg_ptr, point_cloud, _depth_camera_transform);
      point_cloud.header.frame_id = "/map";
      point_cloud.width  = point_cloud_msg_ptr->width;
      point_cloud.height = point_cloud_msg_ptr->height;
      point_cloud.is_dense = false;
      return true;
    }
  }

  bool receiveLogicalImageMsg(const std::string& topic, float duration, LogicalCameraImage& logical_image){
    boost::shared_ptr<LogicalCameraImage const> logical_image_msg_ptr;
    logical_image_msg_ptr = ros::topic::waitForMessage<LogicalCameraImage> (topic, ros::Duration (duration));
    if(logical_image_msg_ptr == NULL){
      ROS_ERROR ("No logical image message received!!!");
      return false;
    }else{
      logical_image = *logical_image_msg_ptr;
      return true;
    }
  }

  void modelBoundingBox(const geometry_msgs::Vector3& min,
			const geometry_msgs::Vector3& max,
			const Eigen::Isometry3f& model_transform,
			pcl::PointXYZ& min_pt,
			pcl::PointXYZ& max_pt){
    
    float x_range = max.x-min.x;
    float y_range = max.y-min.y;
    float z_range = max.z-min.z;

    PointCloud::Ptr model_cloud (new PointCloud ());
    for(int kk=0; kk <= 1; kk++)
      for(int jj=0; jj <= 1; jj++)
	for(int ii=0; ii <= 1; ii++){
	  model_cloud->points.push_back (pcl::PointXYZ(min.x + ii*x_range,
						       min.y + jj*y_range,
						       min.z + kk*z_range));
	}

    PointCloud::Ptr transformed_model_cloud (new PointCloud ());
    
    pcl::transformPointCloud (*model_cloud, *transformed_model_cloud, model_transform);

    pcl::getMinMax3D(*transformed_model_cloud,min_pt,max_pt);

  }

  void filterPointCloud(const PointCloud::ConstPtr& point_cloud,
			const pcl::PointXYZ& min_pt,
			const pcl::PointXYZ& max_pt,
			PointCloud::Ptr& cloud_filtered_xyz){
    
    PointCloud::Ptr cloud_filtered_x (new PointCloud ());
    PointCloud::Ptr cloud_filtered_xy (new PointCloud ());
    
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud (point_cloud);
    pass.setFilterFieldName ("x");
    pass.setFilterLimits (min_pt.x,max_pt.x);
    pass.filter (*cloud_filtered_x);
	    
    pass.setInputCloud (cloud_filtered_x);
    pass.setFilterFieldName ("y");
    pass.setFilterLimits (min_pt.y,max_pt.y);
    pass.filter (*cloud_filtered_xy);
	    
    pass.setInputCloud (cloud_filtered_xy);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (min_pt.z,max_pt.z);
    pass.filter (*cloud_filtered_xyz);

  }


};

int main(int argc, char** argv){
  ros::init(argc,argv,"training_set_generator");
  TrainingSetGenerator generator;
  generator.generate();
  ros::spin();
  return 0;
}
