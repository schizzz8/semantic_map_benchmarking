#include "ros/ros.h"
#include <string>

#include "sensor_msgs/Joy.h"
#include "semantic_map_benchmarking/LogicalCameraImage.h"
#include "semantic_map_extraction/Object.h"
#include "std_msgs/String.h"

#include "geometry_msgs/Vector3.h"
#include "geometry_msgs/Quaternion.h"
#include "tf/tf.h"
#include "tf/transform_datatypes.h"
#include "tf/transform_listener.h"

#include "Eigen/Dense"

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/common/transforms.h>

#include <tf/transform_broadcaster.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>


using namespace std;
using namespace message_filters;
using namespace sensor_msgs;
using namespace semantic_map_benchmarking;


class BoundingBoxVisualizer{
public:
  BoundingBoxVisualizer(std::string robotname_ = "")
    :_robotname(robotname_),
     _logical_image_sub(_nh,"/gazebo/logical_camera_image",1),
     _depth_cloud_sub(_nh,"/camera/depth/points",1),
     _synchronizer(FilterSyncPolicy(10),_logical_image_sub,_depth_cloud_sub)
  {

    //_logical_image_sub = _nh.subscribe("gazebo/logical_camera_image", 1000, &BoundingBoxVisualizer::logicalImageCallback,this);
    _synchronizer.registerCallback(boost::bind(&BoundingBoxVisualizer::filterCallback, this, _1, _2));

    _boxes_cloud_pub = _nh.advertise<sensor_msgs::PointCloud2>("/bounding_boxes_cloud",1);

    ROS_INFO("Starting bounding_box_visualizer_node!");

  }

  void filterCallback(const semantic_map_benchmarking::LogicalCameraImage::ConstPtr& logical_image_msg,
		      const sensor_msgs::PointCloud2::ConstPtr& scene_cloud_msg){

    sensor_msgs::PointCloud2 boxes_cloud_msg;
    boxes_cloud_msg.header.stamp = ros::Time::now();
    boxes_cloud_msg.header.frame_id = "/map";
    boxes_cloud_msg.height = logical_image_msg->models.size()*8;
    boxes_cloud_msg.width  = 1;
    boxes_cloud_msg.is_dense = false;
    sensor_msgs::PointCloud2Modifier pcd_modifier(boxes_cloud_msg);
    pcd_modifier.setPointCloud2FieldsByString(1, "xyz");
	    
    sensor_msgs::PointCloud2Iterator<float> iter_x(boxes_cloud_msg, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(boxes_cloud_msg, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(boxes_cloud_msg, "z");

    tf::StampedTransform logical_camera_pose;
    tf::poseMsgToTF(logical_image_msg->pose,logical_camera_pose);

    for(int i=0; i < logical_image_msg->models.size(); i++){

      tf::Transform model_pose;
      tf::poseMsgToTF(logical_image_msg->models.at(i).pose,model_pose);
          
      Eigen::Vector3f min = tfTransform2eigen(logical_camera_pose)*tfTransform2eigen(model_pose)*
	Eigen::Vector3f(logical_image_msg->models.at(i).min.x,
			logical_image_msg->models.at(i).min.y,
			logical_image_msg->models.at(i).min.z);

      Eigen::Vector3f max = tfTransform2eigen(logical_camera_pose)*tfTransform2eigen(model_pose)*
	Eigen::Vector3f(logical_image_msg->models.at(i).max.x,
			logical_image_msg->models.at(i).max.y,
			logical_image_msg->models.at(i).max.z);

      float x_range = max.x()-min.x();
      float y_range = max.y()-min.y();
      float z_range = max.z()-min.z();

      for(int k=0; k <= 1; k++)
	for(int j=0; j <= 1; j++)
	  for(int i=0; i <= 1; i++,++iter_x,++iter_y,++iter_z){
	    *iter_x = min.x() + i*x_range;
	    *iter_y = min.y() + j*y_range;
	    *iter_z = min.z() + k*z_range;
	  }
    }
    _boxes_cloud_pub.publish(boxes_cloud_msg);

  }

private:
  ros::NodeHandle _nh;
  std::string _robotname = "";

  //ros::Subscriber _logical_image_sub;
  message_filters::Subscriber<LogicalCameraImage> _logical_image_sub;
  message_filters::Subscriber<PointCloud2> _depth_cloud_sub;
  typedef sync_policies::ApproximateTime<LogicalCameraImage,PointCloud2> FilterSyncPolicy;
  message_filters::Synchronizer<FilterSyncPolicy> _synchronizer;

  tf::TransformListener _listener;

  ros::Publisher _objects_cloud_pub;
  ros::Publisher _boxes_cloud_pub;

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

  ros::init(argc, argv, "bounding_box_visualizer");

  BoundingBoxVisualizer bounding_box_visualizer;

  ros::spin();

  return 0;
}
