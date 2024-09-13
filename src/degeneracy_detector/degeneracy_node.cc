#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"
#include "nav_msgs/Odometry.h"
#include "pcl_conversions/pcl_conversions.h"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include "degeneracy_detector/degeneracy_detector.h"
#include <tf/tf.h> 
#include <string>

class DegeneracyNode {
public:
    DegeneracyNode(size_t buffer_size)
        : detector_(buffer_size), map_cloud_(new pcl::PointCloud<pcl::PointXYZ>()) {
        ros::NodeHandle nh;
        // Initialize the subscriber for LIDAR point clouds
        sub_ = nh.subscribe("/velodyne_points", 1, &DegeneracyNode::cloudCallback, this);
        // Initialize the subscriber for odometry (pose)
        odom_sub_ = nh.subscribe("odom", 1, &DegeneracyNode::odomCallback, this);
    }

    void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*cloud_msg, *cloud);

        // If map_cloud_ is empty, use the first received cloud as the map
        if (map_cloud_->empty()) {
            *map_cloud_ = *cloud;
            ROS_INFO("Map cloud initialized.");
            return;  // Skip degeneracy detection on the first cloud
        }

        // Construct the pose vector
        Eigen::Matrix<double, 6, 1> current_pose;
        current_pose << current_pose_x_, current_pose_y_, current_pose_z_, current_roll_, current_pitch_, current_yaw_;

        if (detector_.detectDegeneracy(cloud, map_cloud_, current_pose)) {
            ROS_WARN("Degeneracy detected!");
        } else {
            ROS_INFO("No degeneracy detected.");
        }
    }

    void odomCallback(const nav_msgs::Odometry::ConstPtr& odom_msg) {
        // Extract the position and orientation (quaternion)
        current_pose_x_ = odom_msg->pose.pose.position.x;
        current_pose_y_ = odom_msg->pose.pose.position.y;
        current_pose_z_ = odom_msg->pose.pose.position.z;

        // Convert the quaternion to roll, pitch, and yaw
        tf::Quaternion q(
            odom_msg->pose.pose.orientation.x,
            odom_msg->pose.pose.orientation.y,
            odom_msg->pose.pose.orientation.z,
            odom_msg->pose.pose.orientation.w);
        tf::Matrix3x3 m(q);
        m.getRPY(current_roll_, current_pitch_, current_yaw_);
    }

private:
    ros::Subscriber sub_;
    ros::Subscriber odom_sub_;
    DegeneracyDetector detector_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr map_cloud_;

    // Current pose values
    double current_pose_x_;
    double current_pose_y_;
    double current_pose_z_;
    double current_roll_;
    double current_pitch_;
    double current_yaw_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "degeneracy_node");

    // Set a buffer size, adjust this based on the desired sensitivity
    size_t buffer_size = 10;

    DegeneracyNode node(buffer_size);
    ROS_INFO("Degeneracy Node initialized and subscribing");
    ros::spin();
    return 0;
}
