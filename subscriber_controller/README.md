# Subscriber controller

This example was developed for the [Memmo project](https://www.memmo-project.eu/) to run the MPC controller in a separate and more powerful PC. 

The subscriber controller is a global controller that runs at 2Khz. It publishes the actual state of the robot `actual_base_state`, `actual_js_state` and sends the reference received by an external publisher to the motors. If no reference is received, it sends the previous one. At the beginning, when no references have been yet received, it reads the actual torque values in the load process and performs impedance control to maintain the current position.

To launch it in the the PAL physics simulator with actuators run the following commands:
```
$ roslaunch talos_pal_physics_simulator talos_pal_physics_simulator_with_actuators.launch
```
```
$ roslaunch talos_controller_configuration default_controllers.launch
```
```
$ roslaunch subscriber_controller subscriber_controller.launch simulation:=true
```

> Disclaimer: This example has not been tested on the real robot yet, so please consider it just as an example.

In this example the user can learn how to:
- Use RT-mutexes to avoiding writing and reading the same data from different threads.
- Use the Dynamic reconfigure API from PAL to tune the impedance gains. You will need to open a `rqt_reconfgure` to tune these gains.
- Register variables using a `rosbag` to visualize it later using `PlotJuggler`.
- Use timers to compute elapsed time or periodicity.
- Use Real-time publishers.
- Configure the global controller for different local joint control types.

Using this example we recommend to the user to change the default control types to `direct_torque_control` or `direct_position_control` and try to run this example in Gazebo or in the PAL physics simulator without actuators.

### Threading

This guide is intended to exemplify how to perform the computations in a separate thread from the control thread without using the `pal_base_ros_controller` interface. 

It is very important not to perform RT unsafe operations (e.g. system calls, new/delete, mutex lock) in the controller update call.

To do that, include the thread as a member of your controller:

```c
boost::thread control_thread_data_thread_;
```

And initialize the thread in the load method:

```c
control_thread_data_thread_ =
boost::thread(&MyController::threadLoopFunction, this);
```

If you want to give put your thread in the RealTime scheduler, do the following in the configure method as well:

```c
struct sched_param params;
params.sched_priority = 40;
if (0 != pthread_setschedparam(control_thread_data_thread_.native_handle(), SCHED_FIFO, &params))
{
  ROS_ERROR("Could not adjust priority of the controller thread.");
  return (false);
}
```

Now this thread will be running at the same time as your controller. The user will need to find ways to share information between these two threads without locking the main controller thread.

Typically `std::mutex::lock()` is not used in the main thread, but `std::mutex::try_lock()` or similar, which are non blocking. Programming RT safe code is quite complex task. Put any non-RT Safe code on the other thread, and try to keep RT safe code to the minimum on the main thread.
