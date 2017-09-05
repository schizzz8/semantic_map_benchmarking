#include "ros/ros.h"

#include "sensor_msgs/Joy.h"
#include <sm_simulation_robot/LogicalCameraImage.h>
#include "semantic_map_extraction/Obs.h"
#include "std_msgs/String.h"

#include "tf/tf.h"

using namespace std;

bool enable = false;
string robotname = "";

ros::Publisher obs_pub;
ros::Publisher label_pub;

tf::Transform gazebo_transform;

void joyCallback(const sensor_msgs::Joy::ConstPtr& msg) {
    if(msg->buttons[0] == 1){
        ROS_INFO("Capturing image!");
        enable = true;
    }
}

void logicalImageCallback(const sm_simulation_robot::LogicalCameraImage::ConstPtr& msg){
    if(enable){

        cerr << "Detected models: " << endl;
        int number_of_models = msg->models.size();
        for(int i=0; i < number_of_models; i++){

            semantic_map_extraction::Obs observation;
            observation.stamp = ros::Time::now();

            tf::Transform model_gazebo_pose;
            tf::poseMsgToTF(msg->models.at(i).pose,model_gazebo_pose);

            tf::Transform model_map_pose = gazebo_transform*model_gazebo_pose;

            observation.posx = model_map_pose.getOrigin().x();
            observation.posy = model_map_pose.getOrigin().y();
            observation.theta = tf::getYaw(model_map_pose.getRotation());

            observation.dimx = msg->models.at(i).size.x;
            observation.dimy = msg->models.at(i).size.y;
            observation.dimz = msg->models.at(i).size.z;

            observation.properties = "color:red";

            obs_pub.publish(observation);

            std_msgs::String label;
            label.data = "(\"\\" + msg->models.at(i).type + "\")";
            label_pub.publish(label);

            cerr << msg->models.at(i).type << endl;

        }

        enable = false;
    }
}

int main(int argc, char **argv) {

    ros::init(argc, argv, "ground_truth_publisher");
    ros::NodeHandle nh;

    string joy_topic;
    nh.param<std::string>("joy_topic", joy_topic, "joy");
    ros::Subscriber joy_sub = nh.subscribe(joy_topic, 1000, joyCallback);

    string logical_image_topic;
    nh.param<std::string>("logical_image_topic", logical_image_topic, "gazebo/logical_camera_image");
    ros::Subscriber logical_image_sub = nh.subscribe(logical_image_topic, 1000, logicalImageCallback);

    string obs_topic;
    nh.param<std::string>("obs_topic", obs_topic, "/ground_truth/ground_truth_observation");
    obs_pub = nh.advertise<semantic_map_extraction::Obs>(obs_topic, 1000);

    string label_topic;
    nh.param<std::string>("label_topic", label_topic, "/ground_truth/ground_truth_name");
    label_pub = nh.advertise<std_msgs::String>(label_topic, 1000);

    gazebo_transform.setOrigin(tf::Vector3 (0.059,0.12,0));
    gazebo_transform.getRotation().setRPY(0,0,-0.023);

    ROS_INFO("Starting ground_truth_publisher_node!");

    ros::spin();

    return 0;
}
