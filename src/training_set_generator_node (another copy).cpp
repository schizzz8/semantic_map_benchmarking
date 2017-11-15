#include "ros/ros.h"
#include <string>
#include <sstream>
#include <fstream>

#include "srrg_types/cloud_3d.h"

#include "Eigen/Dense"

#include <sensor_msgs/CameraInfo.h>

#include "sensor_msgs/Joy.h"
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include "semantic_map_benchmarking/LogicalCameraImage.h"

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include "geometry_msgs/Vector3.h"
#include "geometry_msgs/Quaternion.h"
#include "tf/tf.h"
#include "tf/transform_datatypes.h"
#include "tf/transform_listener.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace message_filters;
using namespace sensor_msgs;
using namespace semantic_map_benchmarking;
using namespace srrg_core;

const float low=-std::numeric_limits<int>::max();
const float up=std::numeric_limits<int>::max();

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

class TrainingSetGenerator{
public:
    TrainingSetGenerator(std::string robotname_ = ""):
        _robotname(robotname_),
        _joy_sub(_nh,"/joy",1),
        _rgb_image_sub(_nh,"/camera/rgb/image_raw", 1),
        _dpt_image_sub(_nh,"/camera/depth/image_raw", 1),
        _logical_image_sub(_nh,"gazebo/logical_camera_image", 1),
        _synchronizer(FilterSyncPolicy(10),_joy_sub,_rgb_image_sub,_dpt_image_sub,_logical_image_sub)
    {
        _raw_depth_scale = 0.001;

        _got_info = false;
        _seq = 1;

        _camera_info_subscriber = _nh.subscribe("/camera/depth/camera_info",
                                                1000,
                                                &TrainingSetGenerator::cameraInfoCallback,
                                                this);

        _synchronizer.registerCallback(boost::bind(&TrainingSetGenerator::filterCallback, this, _1, _2, _3, _4));
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

        _invK = _K.inverse();

        _got_info = true;
        _camera_info_subscriber.shutdown();

    }

  void filterCallback(const sensor_msgs::Joy::ConstPtr& joy_msg,
		      const sensor_msgs::Image::ConstPtr& rgb_image_msg,
		      const sensor_msgs::Image::ConstPtr& dpt_image_msg,
		      const semantic_map_benchmarking::LogicalCameraImage::ConstPtr& logical_image_msg){

    if(_got_info && joy_msg->buttons[0] == 1){
      ROS_INFO("Capturing image!");

      cv_bridge::CvImageConstPtr rgb_cv_ptr;
      try{
	rgb_cv_ptr = cv_bridge::toCvShare(rgb_image_msg);
      } catch (cv_bridge::Exception& e) {
	ROS_ERROR("cv_bridge exception: %s", e.what());
	return;
      }

      stringstream ss;
      ss << std::setw(6) << std::setfill('0') << _seq;
      string filename = ss.str();
      cv::imwrite(filename+".png",rgb_cv_ptr->image.clone());

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

      cv_bridge::CvImageConstPtr dpt_cv_ptr;
      try {
	dpt_cv_ptr =  cv_bridge::toCvShare(dpt_image_msg);
      } catch (cv_bridge::Exception& e) {
	ROS_ERROR("cv_bridge exception: %s", e.what());
      }

      cv::Mat depth_image;
      dpt_cv_ptr->image.convertTo(depth_image,CV_16UC1,1000);

      int rows = depth_image.rows;
      int cols = depth_image.cols;
      int num_points = rows*cols;

      Cloud3D* scene_cloud = new Cloud3D;
      scene_cloud->resize(num_points);

      for (int r=0; r<rows; r++) {
	const unsigned short* id_ptr  = depth_image.ptr<unsigned short>(r);
	for (int c=0; c<cols; c++,id_ptr++) {
	  unsigned short id = *id_ptr;
	  float d = id * _raw_depth_scale;
	  Eigen::Vector3f point = tfTransform2eigen(depth_camera_pose) * _invK * Eigen::Vector3f(c*d,r*d,d);
	  scene_cloud->at(c+r*cols) = RichPoint3D(point);
	}
      }

      ofstream file;
      file.open(filename+".txt");
      int number_of_models = logical_image_msg->models.size();

      cv::Mat temp;
      temp = rgb_cv_ptr->image.clone();

      tf::StampedTransform logical_camera_pose;
      tf::poseMsgToTF(logical_image_msg->pose,logical_camera_pose);

      for(int i=0; i < number_of_models; i++){

	std::string object_type = logical_image_msg->models.at(i).type;
	std::string object = object_type.substr(0,object_type.find_first_of("_"));

	Eigen::Vector2i min(up,up);
	Eigen::Vector2i max(low,low);

	tf::Transform model_pose;
	tf::poseMsgToTF(logical_image_msg->models.at(i).pose,model_pose);
		
	Eigen::Isometry3f offset = Eigen::Isometry3f::Identity();
	//offset.translate(Eigen::Vector3f(-0.087,0.0475,1.5));
	offset.rotate(Eigen::Quaternionf(0.5,-0.5,0.5,-0.5));

	Eigen::Vector3f model_min = tfTransform2eigen(logical_camera_pose)*
	  tfTransform2eigen(model_pose)*
	  Eigen::Vector3f(logical_image_msg->models.at(i).min.x,
			  logical_image_msg->models.at(i).min.y,
			  logical_image_msg->models.at(i).min.z);
	Eigen::Vector3f model_max = offset*
	  tfTransform2eigen(model_pose)*
	  Eigen::Vector3f(logical_image_msg->models.at(i).max.x,
			  logical_image_msg->models.at(i).max.y,
			  logical_image_msg->models.at(i).max.z);


	int count=0;
	for(int j=0; j<num_points; j++){

	  Eigen::Vector3f scene_point = scene_cloud->at(j).point();

	  if(scene_point.x() > model_min.x() && scene_point.x() < model_max.x() &&
	     scene_point.y() > model_min.y() && scene_point.y() < model_max.y() &&
	     scene_point.z() > model_min.z() && scene_point.z() < model_max.z())
	    count++;

	}
	cerr << count << endl;

	file.close();

	ofstream outfile;
	outfile.open("depth.cloud");
	scene_cloud->write(outfile);
	outfile.close();
      }

    }

  }

private:
    ros::NodeHandle _nh;
    std::string _robotname = "";
    float _raw_depth_scale;

    bool _got_info;
    Eigen::Matrix3f _K;
    Eigen::Matrix3f _invK;
    int _seq;

    tf::TransformListener _listener;

    ros::Subscriber _camera_info_subscriber;

    message_filters::Subscriber<Joy> _joy_sub;
    message_filters::Subscriber<Image> _rgb_image_sub;
    message_filters::Subscriber<Image> _dpt_image_sub;
    message_filters::Subscriber<LogicalCameraImage> _logical_image_sub;
    typedef sync_policies::ApproximateTime<Joy, Image, Image, LogicalCameraImage> FilterSyncPolicy;
    message_filters::Synchronizer<FilterSyncPolicy> _synchronizer;



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

};


int main(int argc, char **argv) {

    ros::init(argc, argv, "training_set_generator");

    TrainingSetGenerator generator;

    ros::spin();

    return 0;
}
