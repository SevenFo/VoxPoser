#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <geometry_msgs/msg/pose.hpp>
#include "hello_moveit_action/action/move_and_gripper.hpp"  // Custom action

using MoveAndGripper = hello_moveit_action::action::MoveAndGripper;
using GoalHandleMoveAndGripper = rclcpp_action::ServerGoalHandle<MoveAndGripper>;

static const rclcpp::Logger LOGGER = rclcpp::get_logger("move_group_demo");

void setupMoveGroup(moveit::planning_interface::MoveGroupInterface& move_group) {
  RCLCPP_INFO(LOGGER, "Planning frame: %s", move_group.getPlanningFrame().c_str());
  RCLCPP_INFO(LOGGER, "End effector link: %s", move_group.getEndEffectorLink().c_str());
  RCLCPP_INFO(LOGGER, "Available Planning Groups:");
  std::copy(move_group.getJointModelGroupNames().begin(), move_group.getJointModelGroupNames().end(),
            std::ostream_iterator<std::string>(std::cout, ", "));
}

bool planPoseGoal(moveit::planning_interface::MoveGroupInterface& move_group, geometry_msgs::msg::Pose target_pose) {
  move_group.setPoseTarget(target_pose);
  moveit::planning_interface::MoveGroupInterface::Plan my_plan;
  bool success = (move_group.plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS);
  RCLCPP_INFO(LOGGER, "Visualizing plan (pose goal) %s", success ? "" : "FAILED");
  if (success) {
    move_group.move();
  }
  return success;
}

void detachAndRemoveObjects(moveit::planning_interface::PlanningSceneInterface& planning_scene_interface, moveit::planning_interface::MoveGroupInterface& move_group, const std::string& object_id) {
  RCLCPP_INFO(LOGGER, "Detach the object from the robot");
  move_group.detachObject(object_id);

  std::vector<std::string> object_ids = {object_id, "box1"};
  planning_scene_interface.removeCollisionObjects(object_ids);
  RCLCPP_INFO(LOGGER, "Remove the objects from the world");
}

class MoveGroupActionServer : public rclcpp::Node {
public:
  MoveGroupActionServer(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
      : Node("move_group_action_server", options),
        move_group_(std::make_shared<rclcpp::Node>("move_group_interface_tutorial"), "panda_arm"),
        planning_scene_interface_() {
    joint_state_publisher_ = this->create_publisher<sensor_msgs::msg::JointState>("/isaac_joint_commands", 10);
    this->action_server_ = rclcpp_action::create_server<MoveAndGripper>(
        this,
        "move_and_gripper",
        std::bind(&MoveGroupActionServer::handle_goal, this, std::placeholders::_1, std::placeholders::_2),
        std::bind(&MoveGroupActionServer::handle_cancel, this, std::placeholders::_1),
        std::bind(&MoveGroupActionServer::handle_accepted, this, std::placeholders::_1));
    setupMoveGroup(move_group_);

  }

private:
  rclcpp_action::Server<MoveAndGripper>::SharedPtr action_server_;
  moveit::planning_interface::MoveGroupInterface move_group_;
  moveit::planning_interface::PlanningSceneInterface planning_scene_interface_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_publisher_;

  rclcpp_action::GoalResponse handle_goal(const rclcpp_action::GoalUUID& uuid, std::shared_ptr<const MoveAndGripper::Goal> goal) {
    RCLCPP_INFO(this->get_logger(), "Received goal request with gripper position: %.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f, gripper state: %d", goal->pose_goal[0],goal->pose_goal[1],goal->pose_goal[2],goal->pose_goal[3],goal->pose_goal[4],goal->pose_goal[5],goal->pose_goal[6],goal->gripper_state);
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse handle_cancel(const std::shared_ptr<GoalHandleMoveAndGripper> goal_handle) {
    RCLCPP_INFO(this->get_logger(), "Received request to cancel goal");
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handle_accepted(const std::shared_ptr<GoalHandleMoveAndGripper> goal_handle) {
    using namespace std::placeholders;
    std::thread{std::bind(&MoveGroupActionServer::execute, this, _1), goal_handle}.detach();
  }

  void controlGripper(int gripper_state) {
    // Create a JointState message
    auto joint_state_msg = std::make_shared<sensor_msgs::msg::JointState>();

    // Set the target positions for specific joints
    if (gripper_state == 1) {
        RCLCPP_INFO(this->get_logger(), "Opening gripper");
        joint_state_msg->name = {"panda_finger_joint1", "panda_finger_joint2"};
        joint_state_msg->position = {0.04, 0.04};  // Open position
    } else {
        RCLCPP_INFO(this->get_logger(), "Closing gripper");
        joint_state_msg->name = {"panda_finger_joint1", "panda_finger_joint2"};
        joint_state_msg->position = {0.0, 0.0};  // Closed position
    }

    // Publish the joint state
    joint_state_publisher_->publish(*joint_state_msg);
  }


  void execute(const std::shared_ptr<GoalHandleMoveAndGripper> goal_handle) {
    RCLCPP_INFO(this->get_logger(), "Executing goal");
    const auto goal = goal_handle->get_goal();
    auto result = std::make_shared<MoveAndGripper::Result>();

    // Extract pose from the goal
    geometry_msgs::msg::Pose target_pose;
    target_pose.position.x = goal->pose_goal[0];
    target_pose.position.y = goal->pose_goal[1];
    target_pose.position.z = goal->pose_goal[2];
    target_pose.orientation.x = goal->pose_goal[3];
    target_pose.orientation.y = goal->pose_goal[4];
    target_pose.orientation.z = goal->pose_goal[5];
    target_pose.orientation.w = goal->pose_goal[6];

    // Plan and move to the pose goal
    bool success = planPoseGoal(move_group_, target_pose);

    // Control the gripper based on the state
    this->controlGripper(goal->gripper_state);

    if (success) {
      RCLCPP_INFO(this->get_logger(), "Goal succeeded");
      goal_handle->succeed(result);
    } else {
      RCLCPP_INFO(this->get_logger(), "Goal failed");
      goal_handle->abort(result);
    }
  }
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  // 创建参数容器
  rclcpp::NodeOptions options;
  options.parameter_overrides().emplace_back("use_sim_time", rclcpp::ParameterValue(true));
  auto action_server_node = std::make_shared<MoveGroupActionServer>(options);
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(action_server_node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
