^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package kangaroo_pal_physics_simulator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Forthcoming
-----------
* Merge branch 'closed_kinematic_chains' into 'master'
  added the KangarooPhysicsSimulator class and export it as plugin
  See merge request control/kangaroo_pal_physics_simulator!2
* added missing install rules
* fix the launch files to use type and get pid gains argument
* added fixed base PID gains for all joints
* correct the formulations and finish the formulation of the fixed links inverse generalized inertia computation
* parse the right simulator type through launch files
* code cleanup for the new changes in the pal_physics_simulator
* added the KangarooPhysicsSimulator class and export it as plugin
* Contributors: Adria Roig, Sai Kishor Kothakota

0.0.3 (2021-09-08)
------------------
* add pal_geometry_tools to the catkin components
* Contributors: Sai Kishor Kothakota

0.0.2 (2021-09-07)
------------------
* added pal_geometry_tools depedency
* Contributors: Sai Kishor Kothakota

0.0.1 (2021-08-31)
------------------
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
