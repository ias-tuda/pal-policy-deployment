// Copyright 2021 PAL Robotics SL.
// All Rights Reserved
//
// Unauthorized copying of this file, via any medium is strictly prohibited,
// unless it was supplied under the terms of a license agreement or
// nondisclosure agreement with PAL Robotics SL. In this case it may not be
// copied or disclosed except in accordance with the terms of that agreement.
#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <kangaroo_leg_specifics/kangaroo_leg_specifics.h>
#include <pal_utils/permutation.h>

namespace pal
{
class KangarooExtendedJointStatesPublisher
{
public:
  KangarooExtendedJointStatesPublisher()
  {
    kls::LegSpecificsParams leg_params;
    leg_params.readConfig<ariles::ros>(nh_, "/leg_params");
    leg_specifics_.reset(new kls::LegSpecifics(leg_params));
    js_sub_ = nh_.subscribe("/simulator/joint_states", 1,
                            &KangarooExtendedJointStatesPublisher::jointStateCb, this);
    extended_js_pub_ =
        nh_.advertise<sensor_msgs::JointState>("/simulator/joint_states_extended", 1, true);
  }

  void jointStateCb(const sensor_msgs::JointStateConstPtr &msg)
  {
    extended_msg_ = *msg;
    for (const std::string &side : { "left", "right" })
    {
      const std::string leg_length_joint_name = "leg_" + side + "_length_joint";
      const std::string femur_joint_name = "leg_" + side + "_femur_joint";
      const std::string knee_joint_name = "leg_" + side + "_knee_joint";
      int index = indexVector<std::string>(msg->name, leg_length_joint_name);
      if (index == -1)
      {
        ROS_ERROR_STREAM("Index of the joint " << leg_length_joint_name << " not found!");
        continue;
      }
      const double leg_len_pos = msg->position[index];
      leg_specifics_->setLegExtension(leg_len_pos);
      extended_msg_.name.push_back(knee_joint_name);
      extended_msg_.position.push_back(leg_specifics_->getKneeAngle());
      extended_msg_.velocity.push_back(0.0);
      extended_msg_.effort.push_back(0.0);
      extended_msg_.name.push_back(femur_joint_name);
      extended_msg_.position.push_back(leg_specifics_->getFemurAngle());
      extended_msg_.velocity.push_back(0.0);
      extended_msg_.effort.push_back(0.0);
      extended_js_pub_.publish(extended_msg_);
    }
  }

private:
  ros::NodeHandle nh_;
  ros::Publisher extended_js_pub_;
  ros::Subscriber js_sub_;
  sensor_msgs::JointState extended_msg_;
  std::unique_ptr<kls::LegSpecifics> leg_specifics_;
};
}

int main(int argc, char *argv[])
{
  ros::init(argc, argv, "extended_joint_states_publisher");
  ros::NodeHandle nh;
  pal::KangarooExtendedJointStatesPublisher publisher;
  ros::spin();
  return 0;
}
