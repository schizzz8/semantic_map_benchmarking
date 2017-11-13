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
#include <pcl/common/common.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/passthrough.h>

#include <tf/transform_broadcaster.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>


using namespace std;
using namespace message_filters;

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

class BoundingBoxVisualizer{
public:
    BoundingBoxVisualizer(std::string robotname_ = "")
        :_robotname(robotname_),
          _logical_image_sub(_nh,"/gazebo/logical_camera_image",1),
          _depth_cloud_sub(_nh,"/camera/depth/points",1),
          _synchronizer(FilterSyncPolicy(10),_logical_image_sub,_depth_cloud_sub) {

        _synchronizer.registerCallback(boost::bind(&BoundingBoxVisualizer::filterCallback, this, _1, _2));

        _boxes_cloud_pub = _nh.advertise<PointCloud>("/bounding_boxes_cloud",10);
        _objects_cloud_pub = _nh.advertise<PointCloud>("/objects_point_cloud",10);

        ROS_INFO("Starting bounding_box_visualizer_node!");

    }

    void filterCallback(const semantic_map_benchmarking::LogicalCameraImage::ConstPtr& logical_image_msg,
                        const PointCloud::ConstPtr& scene_cloud_msg){


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
        
        PointCloud::Ptr objects_cloud_msg (new PointCloud ());

        PointCloud::Ptr boxes_cloud_msg (new PointCloud ());
       
        tf::StampedTransform logical_camera_pose;
        tf::poseMsgToTF(logical_image_msg->pose,logical_camera_pose);

        for(int i=0; i < logical_image_msg->models.size(); i++){

            Eigen::Vector3f box_min (logical_image_msg->models.at(i).min.x,
				     logical_image_msg->models.at(i).min.y,
				     logical_image_msg->models.at(i).min.z);

            Eigen::Vector3f box_max (logical_image_msg->models.at(i).max.x,
				     logical_image_msg->models.at(i).max.y,
				     logical_image_msg->models.at(i).max.z);

            float x_range = box_max.x()-box_min.x();
            float y_range = box_max.y()-box_min.y();
            float z_range = box_max.z()-box_min.z();

	    PointCloud::Ptr model_cloud (new PointCloud ());
            for(int k=0; k <= 1; k++)
	      for(int j=0; j <= 1; j++)
		for(int i=0; i <= 1; i++){
		  model_cloud->points.push_back (pcl::PointXYZ(box_min.x() + i*x_range,
								   box_min.y() + j*y_range,
								   box_min.z() + k*z_range));
		}

	    PointCloud::Ptr transformed_model_cloud (new PointCloud ());
            tf::Transform model_pose;
            tf::poseMsgToTF(logical_image_msg->models.at(i).pose,model_pose);
	    Eigen::Isometry3f model_transform = tfTransform2eigen(logical_camera_pose)*tfTransform2eigen(model_pose);
	    pcl::transformPointCloud (*model_cloud, *transformed_model_cloud, model_transform);

	    *boxes_cloud_msg += *transformed_model_cloud;
	    
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

	    *objects_cloud_msg += *cloud_filtered_xyz;

        }

	boxes_cloud_msg->header.frame_id = "/map";
        boxes_cloud_msg->height = 1;
        boxes_cloud_msg->width  = logical_image_msg->models.size()*8;
        boxes_cloud_msg->is_dense = false;
	_boxes_cloud_pub.publish(boxes_cloud_msg);

	objects_cloud_msg->header.frame_id = "/map";
        objects_cloud_msg->width  = objects_cloud_msg->points.size();
        objects_cloud_msg->height = 1;
	objects_cloud_msg->is_dense = false;
        _objects_cloud_pub.publish(objects_cloud_msg);

    }

private:
    ros::NodeHandle _nh;
    std::string _robotname = "";

    message_filters::Subscriber<semantic_map_benchmarking::LogicalCameraImage> _logical_image_sub;
    message_filters::Subscriber<PointCloud> _depth_cloud_sub;
    typedef sync_policies::ApproximateTime<semantic_map_benchmarking::LogicalCameraImage,PointCloud> FilterSyncPolicy;
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
