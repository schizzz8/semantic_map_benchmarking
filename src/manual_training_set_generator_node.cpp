#include <iostream>
#include <ros/ros.h>
#include "sensor_msgs/Joy.h"
#include <sensor_msgs/CameraInfo.h>
#include <Eigen/Core>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include "semantic_map_benchmarking/LogicalCameraImage.h"
#include "tf/tf.h"
#include "tf/transform_listener.h"
#include "tf/transform_datatypes.h"
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/passthrough.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace std;
using namespace message_filters;
using namespace semantic_map_benchmarking;

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

constexpr unsigned int str2int(const char* str, int h = 0){
  return !str[h] ? 5381 : (str2int(str, h+1) * 33) ^ str[h];
}

int object2id(std::string object){
  switch (str2int(object.c_str())){
  case str2int("chair"):
    return 8;
  case str2int("table"):
    return 10;
  case str2int("plant"):
    return 15;
  case str2int("tv"):
    return 19;
  case str2int("couch"):
    return 17;
  default:
    return 20;

  }

}

class TrainingSetGenerator{
public:
  TrainingSetGenerator(std::string robotname_ = ""):
    _robotname(robotname_),
    _logical_image_sub(_nh,"/gazebo/logical_camera_image",1),
    _depth_cloud_sub(_nh,"/camera/depth/points",1),
    _rgb_image_sub(_nh,"/camera/rgb/image_raw", 1),
    _synchronizer(FilterSyncPolicy(10),_logical_image_sub,_depth_cloud_sub,_rgb_image_sub)//,_it(_nh)
  {

    _got_info = false;
    _camera_info_sub = _nh.subscribe("/camera/depth/camera_info",
				     1000,
				     &TrainingSetGenerator::cameraInfoCallback,
				     this);

    _enable = false;
    _joy_sub = _nh.subscribe("/joy", 1000, &TrainingSetGenerator::joyCallback,this);

    _synchronizer.registerCallback(boost::bind(&TrainingSetGenerator::filterCallback, this, _1, _2, _3));

    _seq = 1;

    //_label_image_pub = _it.advertise("/camera/rgb/label_image", 1);
    
    ROS_INFO("Starting training set generator node!");
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
  
  void joyCallback(const sensor_msgs::Joy::ConstPtr& msg) {
    if(msg->buttons[0] == 1){
      ROS_INFO("Capturing image!");
      _enable = true;
    }
  }

  void filterCallback(const semantic_map_benchmarking::LogicalCameraImage::ConstPtr& logical_image_msg,
		      const PointCloud::ConstPtr& scene_cloud_msg,
		      const sensor_msgs::Image::ConstPtr& rgb_image_msg){
    if(_got_info && _enable && !logical_image_msg->models.empty()){

      cv_bridge::CvImageConstPtr rgb_cv_ptr;
      try{
	rgb_cv_ptr = cv_bridge::toCvShare(rgb_image_msg);
      } catch (cv_bridge::Exception& e) {
	ROS_ERROR("cv_bridge exception: %s", e.what());
	return;
      }

      cv::Mat rgb_image = rgb_cv_ptr->image.clone();
      int rows = rgb_image.rows;
      int cols = rgb_image.cols;
      
      //save image
      stringstream ss;
      ss << std::setw(6) << std::setfill('0') << _seq;
      string filename = ss.str();
      cv::imwrite(filename+".jpg",rgb_image);
      cerr << "Saving image: " << filename+".jpg" << endl;

      //open file to store detected objects
      ofstream file;
      file.open(filename+".txt");

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
      PointCloud::Ptr map_cloud (new PointCloud ());
      pcl::transformPointCloud (*scene_cloud_msg, *map_cloud, depth_camera_transform);
      map_cloud->header.frame_id = "/map";
      map_cloud->width  = scene_cloud_msg->width;
      map_cloud->height = scene_cloud_msg->height;
      map_cloud->is_dense = false;
        
      //PointCloud::Ptr objects_cloud_msg (new PointCloud ());
      //PointCloud::Ptr boxes_cloud_msg (new PointCloud ());
       
      tf::StampedTransform logical_camera_pose;
      tf::poseMsgToTF(logical_image_msg->pose,logical_camera_pose);

      for(int idx=0; idx < logical_image_msg->models.size(); idx++){

	Eigen::Vector3f box_min (logical_image_msg->models.at(idx).min.x,
				 logical_image_msg->models.at(idx).min.y,
				 logical_image_msg->models.at(idx).min.z);

	Eigen::Vector3f box_max (logical_image_msg->models.at(idx).max.x,
				 logical_image_msg->models.at(idx).max.y,
				 logical_image_msg->models.at(idx).max.z);

	float x_range = box_max.x()-box_min.x();
	float y_range = box_max.y()-box_min.y();
	float z_range = box_max.z()-box_min.z();

	PointCloud::Ptr model_cloud (new PointCloud ());
	for(int kk=0; kk <= 1; kk++)
	  for(int jj=0; jj <= 1; jj++)
	    for(int ii=0; ii <= 1; ii++){
	      model_cloud->points.push_back (pcl::PointXYZ(box_min.x() + ii*x_range,
							   box_min.y() + jj*y_range,
							   box_min.z() + kk*z_range));
	    }

	PointCloud::Ptr transformed_model_cloud (new PointCloud ());
	tf::Transform model_pose;
	tf::poseMsgToTF(logical_image_msg->models.at(idx).pose,model_pose);
	Eigen::Isometry3f model_transform = tfTransform2eigen(logical_camera_pose)*tfTransform2eigen(model_pose);
	pcl::transformPointCloud (*model_cloud, *transformed_model_cloud, model_transform);

	//*boxes_cloud_msg += *transformed_model_cloud;
	    
	pcl::PointXYZ min_pt,max_pt;
	pcl::getMinMax3D(*transformed_model_cloud,min_pt,max_pt);
	    
	PointCloud::Ptr cloud_filtered_x (new PointCloud ());
	PointCloud::Ptr cloud_filtered_xy (new PointCloud ());
	PointCloud::Ptr cloud_filtered_xyz (new PointCloud ());

	pcl::PassThrough<pcl::PointXYZ> pass;
	pass.setInputCloud (map_cloud);
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

	if(!cloud_filtered_xyz->points.empty()){
	  string object_type = logical_image_msg->models.at(idx).type;
	  string object = object_type.substr(0,object_type.find_first_of("_"));

	  cv::Point2i p_min(10000,10000);
	  cv::Point2i p_max(-10000,-10000);
	  
	  for(int jdx=0; jdx<cloud_filtered_xyz->points.size(); jdx++){
	    Eigen::Vector3f camera_point = depth_camera_transform.inverse()*
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

	    // cv::circle(rgb_image,
	    // 	       cv::Point2i(r,c),
	    // 	       1,
	    // 	       cv::Scalar(255,0,0));
	  }//for cloud_filtered points
	  float abs_x = (float)(p_min.y+p_max.y)/2.0f;
	  float abs_y = (float)(p_min.x+p_max.x)/2.0f;
	  float abs_w = (float)(p_max.y-p_min.y);
	  float abs_h = (float)(p_max.x-p_min.x);
	  
	  file << object2id(object) << " ";
	  file << abs_x/(float)cols << " " << abs_y/(float)rows << " ";
	  file << abs_w/(float)cols << " " << abs_h/(float)rows << endl;

	  // cv::rectangle(rgb_image,
	  // 		p_min,
	  // 		p_max,
	  // 		cv::Scalar(0,0,255));
	}//if cloud_filtered not empty

	//*objects_cloud_msg += *cloud_filtered_xyz;

      }//for models
      file.close();
      _seq++;
      //sensor_msgs::ImagePtr label_image_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();

      //_label_image_pub.publish(label_image_msg);

      _enable = false;
    }//if got_info & enable & !models.empty

  }

private:
  ros::NodeHandle _nh;
  string _robotname;

  ros::Subscriber _camera_info_sub;
  Eigen::Matrix3f _K;
  bool _got_info;

  ros::Subscriber _joy_sub;
  bool _enable;

  tf::TransformListener _listener;
  
  message_filters::Subscriber<LogicalCameraImage> _logical_image_sub;
  message_filters::Subscriber<PointCloud> _depth_cloud_sub;
  message_filters::Subscriber<sensor_msgs::Image> _rgb_image_sub;
  typedef sync_policies::ApproximateTime<LogicalCameraImage,PointCloud,sensor_msgs::Image> FilterSyncPolicy;
  message_filters::Synchronizer<FilterSyncPolicy> _synchronizer;

  //image_transport::ImageTransport _it;
  //image_transport::Publisher _label_image_pub;
  
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

  tf::Transform eigen2tfTransform(const Eigen::Isometry3f& T){
    Eigen::Quaternionf q(T.linear());
    Eigen::Vector3f t=T.translation();
    tf::Transform tft;
    tft.setOrigin(tf::Vector3(t.x(), t.y(), t.z()));
    tft.setRotation(tf::Quaternion(q.x(), q.y(), q.z(), q.w()));
    return tft;
  }
  
  const float low=-std::numeric_limits<int>::max();
  const float up=std::numeric_limits<int>::max();

};

int main (int argc, char** argv){
  
  ros::init(argc, argv, "manual_training_set_generator");

  TrainingSetGenerator manual_generator;

  ros::spin();
  return 0;
}
