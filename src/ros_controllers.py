class SimpleQuadcopterController:
    def __init__(self, env, config) -> None:
        assert (
            type(env) == VoxPoserPyRepQuadcopterEnv or type(env) == DummyEnv
        ), "env type should be VoxPoserPyRepQuadcopterEnv"
        self.env = env

    def execute(self, movable_obs, waypoint, is_object_centric=False):
        """
        execute a waypoint
        If movable is "end effector", then do not consider object interaction (no dynamics considered)
        If movable is "object", then consider object interaction (use heuristics-based dynamics model)

        :param movable_obs: observation dict of the object to be moved
        :param waypoint: list, [target_xyz, target_rotation, target_velocity, target_gripper], target_xyz is for movable in world frame
        :return: None
        """
        info = dict()
        (
            target_xyz,
            target_rotation,
            target_velocity,
            target_gripper,
        ) = waypoint  # ignore target_rotation and target_waypoint
        assert target_gripper is None and target_rotation is None
        object_centric = is_object_centric
        # move to target pose directly
        if not object_centric:
            target_pose = np.concatenate([target_xyz, [0, 0, 0, 1]])
            result = self.env.apply_action(target_pose)
            info["mp_info"] = 0  # for success
        else:
            raise NotImplementedError(
                "not implement execute when movable is not qudacopter"
            )
        return info


class SimpleROSController:
    def __init__(self, env, config) -> None:
        assert (
            type(env) is VoxPoserROSDroneEnv
        ), "env type should be VoxPoserROSDroneEnv"
        self.env = env
        self.mode = config.mode
        print(f"[controllers.py] using mode: {self.mode}")

    def execute(self, movable_obs, target, is_object_centric=False):
        """
        execute a waypoint or whole traj

        :param movable_obs: observation dict of the object to be moved
        :param waypoint: list, [target_xyz, target_rotation, target_velocity, target_gripper], target_xyz is for movable in world frame
        :param mode: str, "pose" or "velocity"
        :return: None
        """
        object_centric = is_object_centric
        if object_centric:
            raise NotImplementedError(
                "not implement execute when movable is not qudacopter"
            )
        info = dict()
        if self.mode != "traj":
            (
                target_xyz,
                target_rotation,
                target_velocity,
                target_gripper,
            ) = target  # ignore target_rotation and target_waypoint
            assert target_gripper is None and target_rotation is None
            # move to target pose directly
            if self.mode == "pose":
                target_pose = np.concatenate([target_xyz, [0, 0, 0, 1]])
                result = self.env.apply_action(target_pose)
            elif self.mode == "velocity":
                result = self._execute_velocity(movable_obs, target_xyz)
        else:
            # traj
            traj = target
            result = self.env.apply_action(traj, mode="traj")
        info["mp_info"] = 0  # for success
        return info

    def is_finished(self):
        return self.env.is_finished()

    def _execute_velocity(self, movable_obs, target_xyz):
        """
        execute a waypoint
        If movable is "end effector", then do not consider object interaction (no dynamics considered)
        If movable is "object", then consider object interaction (use heuristics-based dynamics model)

        :param movable_obs: observation dict of the object to be moved
        :param waypoint: list, [target_xyz, target_rotation, target_velocity, target_gripper], target_xyz is for movable in world frame
        :return: None
        """
        # use PI to control the drone: calculate the error between the target and the current position,
        # then use PI to control the drone velocity in the x, y, z direction
        current_position = self.env.get_ee_pos()
        error = np.array(target_xyz) - np.array(current_position)
        # close loop control
        # PI controller
        kp = 0.3
        ki = 0.0001
        self.integral = np.zeros(3)
        print(
            f"[controllers.py] start PI control, current position: {current_position[0]:.3},{current_position[1]:.3},{current_position[2]:.3} \
target position: {target_xyz[0]:.3},{target_xyz[1]:.3},{target_xyz[2]:.3} \
error: {np.linalg.norm(error):.3} kp: {kp:.2}, ki: {ki:.2}"
        )
        while np.linalg.norm(error) > 0.1:
            self.integral += error
            velocity = kp * error + ki * self.integral
            # move to target pose directly
            print(
                f"[controllers.py] start PI control, current position: {current_position[0]:.3},{current_position[1]:.3},{current_position[2]:.3} \
target position: {target_xyz[0]:.3},{target_xyz[1]:.3},{target_xyz[2]:.3} \
error: {np.linalg.norm(error):.3},\
velocity: {velocity[0]:.3},{velocity[1]:.3},{velocity[2]:.3}"
            )
            result = self.env.apply_action(velocity, mode=self.mode)
            current_position = self.env.get_ee_pos()
            error = np.array(target_xyz) - np.array(current_position)
        result = self.env.apply_action(np.array([0, 0, 0]), mode=self.mode)  # stop
