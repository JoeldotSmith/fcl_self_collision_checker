# fcl_self_collision_checker

This package provides a self-collision checking utility using the Flexible Collision Library (FCL) for Unitree G1.

## Installation
# Prerequisites
This package relies on on FCL and its required dependencies please see https://github.com/flexible-collision-library/fcl for more.

1. Clone the repository into your ROS 2 workspace:
    ```bash
    git clone https://github.com/your-repo/fcl_self_collision_checker.git
    ```
2. Build the workspace:
    ```bash
    colcon build
    ```
3. Source the workspace:
    ```bash
    source install/setup.bash
    ```

## Usage
The node listens to the /tf topic for data on the joint transforms please get the robot moving in rviz first before checking or see mocopi_ros2 on branch g1 for more.

1. Run the collision checker node:
    ```bash
    ros2 run fcl_self_collision_checker collision_checker
    ```

## Benchmark
n: 5000 | Mean: 2.583 ms | StdDev: 1.329 ms | 95% CI: ±0.037 ms

## Dependencies

- ROS 2 (Only tested on foxy)
- Flexible Collision Library (FCL)

## License
This project is licensed under the [MIT License](LICENSE).
