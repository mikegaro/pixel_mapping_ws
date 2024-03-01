
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <cmath>

int main(int argc, char **argv)
{
  ros::init(argc, argv, "points_and_lines");
  ros::NodeHandle n;
  ros::Publisher marker_pub = n.advertise<visualization_msgs::Marker>("visualization_marker", 0);

  ros::Rate rate(5);
  while (ros::ok())
  {
    visualization_msgs::Marker points;
    points.header.frame_id = "camera_pov_link";
    points.header.stamp = ros::Time::now();
    points.ns = "points_and_lines";
    points.action = visualization_msgs::Marker::ADD;
    points.pose.orientation.w = 1.0;
    points.id = 0;
    points.type = visualization_msgs::Marker::POINTS;
    // POINTS markers use x and y scale for width/height respectively
    points.scale.x = 0.01;
    points.scale.y = 0.01;
    // Points are green
    points.color.g = 1.0f;
    points.color.a = 1.0f;
    // Create the vertices for the points and lines
    for (uint32_t i = 0; i < 100; ++i)
    {
      geometry_msgs::Point p;
      p.x = i/100.0f;
      p.y = 0;
      p.z = 0;
      points.points.push_back(p);
    }
    marker_pub.publish(points);
    rate.sleep();
  }
  return 0;
}