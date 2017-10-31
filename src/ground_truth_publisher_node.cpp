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

using namespace std;

constexpr unsigned int str2int(const char* str, int h = 0){
    return !str[h] ? 5381 : (str2int(str, h+1) * 33) ^ str[h];
}

std::string object2color(std::string object){
    switch (str2int(object.c_str())){
    case str2int("table"):
        return std::string("color:red");
    case str2int("chair"):
        return std::string("color:blue");
    case str2int("bookcase"):
        return std::string("color:green");
    case str2int("couch"):
        return std::string("color:yellow");
    case str2int("cabinet"):
        return std::string("color:cyan");
    case str2int("plant"):
        return std::string("color:magenta");
    default:
        return std::string("color:black");
    }

}

class GroundTruthPublisher{
public:
    GroundTruthPublisher(bool enable_ = false,
                         std::string robotname_ = "")
        :_enable(enable_),_robotname(robotname_){


        _nh.param<std::string>("joy_topic", _joy_topic, "/joy");
        _joy_sub = _nh.subscribe(_joy_topic, 1000, &GroundTruthPublisher::joyCallback,this);


        _nh.param<std::string>("logical_image_topic", _logical_image_topic, "gazebo/logical_camera_image");
        _logical_image_sub = _nh.subscribe(_logical_image_topic, 1000, &GroundTruthPublisher::logicalImageCallback,this);


        _nh.param<std::string>("obj_topic", _obj_topic, "/ObjectTopic");
        _obj_pub = _nh.advertise<semantic_map_extraction::Object>(_obj_topic, 1000);


        ROS_INFO("Starting ground_truth_publisher_node!");

    }

    void joyCallback(const sensor_msgs::Joy::ConstPtr& msg) {
        if(msg->buttons[0] == 1){
            ROS_INFO("Capturing image!");
            _enable = true;
        }
    }

    void logicalImageCallback(const semantic_map_benchmarking::LogicalCameraImage::ConstPtr& msg){
        if(_enable){

            cerr << "Detected models: " << endl;
            int number_of_models = msg->models.size();

            for(int i=0; i < number_of_models; i++){
                semantic_map_extraction::Object obj;
                obj.stamp = ros::Time::now();

                std::string object_type = msg->models.at(i).type;
                cerr << "- " << object_type << endl;
                std::string object = object_type.substr(0,object_type.find_first_of("_"));
                obj.type = "(\"\\" + object + "\")";

                tf::Transform model_pose;
                tf::poseMsgToTF(msg->models.at(i).pose,model_pose);

                tf::StampedTransform logical_camera_transform;
                try {
                    _listener.waitForTransform("map",
                                               "logical_camera_link",
                                               ros::Time(0),
                                               ros::Duration(3));
                    _listener.lookupTransform("map",
                                              "logical_camera_link",
                                              ros::Time(0),
                                              logical_camera_transform);
                }
                catch(tf::TransformException ex) {
                    ROS_ERROR("%s", ex.what());
                }

                tf::Transform world_model_pose = logical_camera_transform*model_pose;


                obj.posx = world_model_pose.getOrigin().x();
                obj.posy = world_model_pose.getOrigin().y();
                double roll, pitch, yaw;
                world_model_pose.getBasis().getRPY(roll, pitch, yaw);
                obj.theta = yaw;

                cerr << "\t>>pose: "
                     << obj.posx << ","
                     << obj.posy << ","
                     << obj.theta << endl;

                obj.dimx = msg->models.at(i).size.x;
                obj.dimy = msg->models.at(i).size.y;
                obj.dimz = msg->models.at(i).size.z;

                cerr << "\t>>size: "
                     << obj.dimx << ","
                     << obj.dimy << ","
                     << obj.dimz << endl;

                obj.properties = object2color(object);

                cerr << "\t>>properties: " << obj.properties << endl;

                _obj_pub.publish(obj);

            }

            _enable = false;
        }
    }

private:
    ros::NodeHandle _nh;
    std::string _joy_topic;
    std::string _logical_image_topic;
    std::string _obj_topic;
    ros::Subscriber _joy_sub;
    ros::Subscriber _logical_image_sub;
    bool _enable = false;
    std::string _robotname = "";

    ros::Publisher _obj_pub;

    tf::TransformListener _listener;
};





int main(int argc, char **argv) {

    ros::init(argc, argv, "ground_truth_publisher");

    GroundTruthPublisher ground_truth_publisher;
    
    ros::spin();

    return 0;
}
