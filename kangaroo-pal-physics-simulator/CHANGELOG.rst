^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package kangaroo_pal_physics_simulator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Forthcoming
-----------
* clean the upload actuator parameters launch file
* added extended joint states publisher node
* tune the PID gains of both legs for the functionality of walking controller in position
* update fixed_base launch file
* remove the default configuration as this is being loaded by the kangaroo_description
* remove the simulation parameters with the configuration of collision_parameters in kangaroo_description
* remove the unused robot argument
* move the imu sensor info to leg as we don't load torso with selective rosparam
* Update changes of selective rosparam loader launch file
* Added FT sensors and the IMU in the simulator
* Disable the self collision handling as this is generating higher torques between joints
* Update the PID values of the joints
* Update default configuration position
* change the default configuration height
* Initial commit
* Contributors: Sai Kishor Kothakota
