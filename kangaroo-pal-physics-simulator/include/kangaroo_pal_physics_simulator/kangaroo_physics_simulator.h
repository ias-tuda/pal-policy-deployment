// Copyright 2021 PAL Robotics SL.
// All Rights Reserved
//
// Unauthorized copying of this file, via any medium is strictly prohibited,
// unless it was supplied under the terms of a license agreement or
// nondisclosure agreement with PAL Robotics SL. In this case it may not be
// copied or disclosed except in accordance with the terms of that agreement.
#ifndef KANGAROO_PHYSICS_SIMULATOR_H
#define KANGAROO_PHYSICS_SIMULATOR_H
/// \author Sai Kishor Kothakota

#include <pal_physics_simulator/physics_simulator.h>
namespace pal_physics_simulator
{
class KangarooPhysicsSimulator : public PhysicsSimulator
{
public:
  virtual ~KangarooPhysicsSimulator();

private:
  void computeRelativeJacobian(const Eigen::VectorXd &Q, unsigned int target_body_id,
                               unsigned int base_body_id, Eigen::MatrixXd &jacobian,
                               bool update_kinematics);

  virtual void computeInverseGeneralizedInertia(Eigen::MatrixXd &inv_generalized_inertia,
                                                size_t fixed_links_size) override;
};
}
#endif  // KANGAROO_PHYSICS_SIMULATOR_H
