#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include "hello_moveit_action/action/move_and_gripper.hpp"  // Custom action
#include <gtest/gtest.h>

using MoveAndGripper = hello_moveit_action::action::MoveAndGripper;

class MoveGroupActionServerTest : public ::testing::Test {
protected:
  void SetUp() override {
    node_ = rclcpp::Node::make_shared("test_move_group_action_server");
    action_client_ = rclcpp_action::create_client<MoveAndGripper>(node_, "move_and_gripper");
    ASSERT_TRUE(action_client_->wait_for_action_server(std::chrono::seconds(5)));
  }

  rclcpp::Node::SharedPtr node_;
  rclcpp_action::Client<MoveAndGripper>::SharedPtr action_client_;
};

TEST_F(MoveGroupActionServerTest, TestSendGoal) {
  auto goal_msg = MoveAndGripper::Goal();
  goal_msg.pose_goal[0] = 1.0;
  goal_msg.pose_goal[1] = 0.2;
  goal_msg.pose_goal[2] = 1.0;
  goal_msg.pose_goal[3] = 0;
  goal_msg.pose_goal[4] = 0;
  goal_msg.pose_goal[5] = 0;
  goal_msg.pose_goal[6] = 1;
  goal_msg.gripper_state = 0.0;

  auto send_goal_options = rclcpp_action::Client<MoveAndGripper>::SendGoalOptions();
  send_goal_options.result_callback = [](const rclcpp_action::ClientGoalHandle<MoveAndGripper>::WrappedResult & result) {
    EXPECT_EQ(result.code, rclcpp_action::ResultCode::SUCCEEDED);
    EXPECT_TRUE(result.result->success);
  };

  auto goal_handle_future = action_client_->async_send_goal(goal_msg, send_goal_options);
  rclcpp::spin_until_future_complete(node_, goal_handle_future);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  rclcpp::init(argc, argv);
  int result = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return result;
}
