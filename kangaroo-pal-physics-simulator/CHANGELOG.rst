^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package kangaroo_pal_physics_simulator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Forthcoming
-----------
* Merge branch 'gui_arg' into 'master'
  Add an argument 'gui' to disable the rviz
  See merge request control/kangaroo_pal_physics_simulator!10
* Apply 1 suggestion(s) to 1 file(s)
* Add an argument 'gui' to disable the rviz
* Contributors: Pierre Fernbach, Sai Kishor Kothakota

0.0.11 (2024-07-12)
-------------------
* Change pal_physics_simulator to depend directly
* Contributors: Sai Kishor Kothakota

0.0.10 (2024-04-19)
-------------------
* Merge branch 'add/urdf/version_argument' into 'master'
  add version argument to the pal physics simulator launch
  See merge request control/kangaroo_pal_physics_simulator!9
* add version argument to the pal physics simulator launch
* Contributors: Adria Roig, Sai Kishor Kothakota

0.0.9 (2023-02-06)
------------------
* Update the rviz config with FT information
* added the missing imu config as selective_rosparam_loader.launch doesn't do it
* Contributors: Sai Kishor Kothakota

0.0.8 (2023-01-17)
------------------
* Merge branch 'ft_robot' into 'master'
  Add argument to start robot with ft_sensors or not
  See merge request control/kangaroo_pal_physics_simulator!7
* unify enable_ft and ft_sensors argument with new ft_sensor_link changes
* Propagate ft_sensors argument to the bringup
* Add argument to start robot with ft_sensors or not
* Merge branch 'added_enable_ft' into 'master'
  added a way to enable/disable the FT sensors in the simulator
  See merge request control/kangaroo_pal_physics_simulator!5
* moved the imu sensor information to torso
* added a way to enable/disable the FT sensors in the simulator
* Contributors: Adria Roig, Adrià Roig, Sai Kishor Kothakota, saikishor

0.0.7 (2022-12-22)
------------------
* Merge branch 'add_use_cage_argument' into 'master'
  add use_cage argument to the launch files
  See merge request control/kangaroo_pal_physics_simulator!6
* add use_cage argument to the launch files
* Contributors: Sai Kishor Kothakota, saikishor

0.0.6 (2022-03-14)
------------------
* Merge branch 'enable_crane' into 'master'
  Enable crane
  See merge request control/kangaroo_pal_physics_simulator!4
* unify the launches and added the proper arguments for the enable_crane part
* Tune more PID gains
* Contributors: Adria Roig, Sai Kishor Kothakota

0.0.5 (2022-03-07)
------------------
* fix missing kangaroo_bringup exec dependency
* Contributors: Sai Kishor Kothakota

0.0.4 (2021-11-17)
------------------
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
