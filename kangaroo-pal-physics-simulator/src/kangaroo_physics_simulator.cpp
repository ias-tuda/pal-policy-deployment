// Copyright 2021 PAL Robotics SL.
// All Rights Reserved
//
// Unauthorized copying of this file, via any medium is strictly prohibited,
// unless it was supplied under the terms of a license agreement or
// nondisclosure agreement with PAL Robotics SL. In this case it may not be
// copied or disclosed except in accordance with the terms of that agreement.
#include <kangaroo_pal_physics_simulator/kangaroo_physics_simulator.h>
#include <math_utils/eigen/qr_factorization.h>
#include <pluginlib/class_list_macros.h>

namespace pal_physics_simulator
{
KangarooPhysicsSimulator::~KangarooPhysicsSimulator()
{
}

void KangarooPhysicsSimulator::computeRelativeJacobian(const Eigen::VectorXd &Q,
                                                       unsigned int target_body_id,
                                                       unsigned int base_body_id,
                                                       Eigen::MatrixXd &jacobian,
                                                       bool update_kinematics)
{
  /* compute base and target jacobians*/
  Eigen::MatrixXd base_jacobian, target_jacobian;
  base_jacobian.resize(jacobian.rows(), jacobian.cols());
  target_jacobian.resize(jacobian.rows(), jacobian.cols());

  /* compute base orientation wrt world */
  Eigen::Matrix3d world_R_base =
      RigidBodyDynamics::CalcBodyWorldOrientation(dynamic_model_.rbdl_model_, Q,
                                                  base_body_id, update_kinematics)
          .transpose();

  /* Compute the vector origin_target-origin_base w.r.t. world */
  Eigen::Vector3d world_origin_base = RigidBodyDynamics::CalcBodyToBaseCoordinates(
      dynamic_model_.rbdl_model_, Q, base_body_id, Eigen::Vector3d::Zero(), update_kinematics);
  Eigen::Vector3d world_origin_target = RigidBodyDynamics::CalcBodyToBaseCoordinates(
      dynamic_model_.rbdl_model_, Q, target_body_id, Eigen::Vector3d::Zero(), update_kinematics);

  RigidBodyDynamics::CalcPointJacobian6D(dynamic_model_.rbdl_model_, Q, base_body_id,
                                         world_R_base.inverse() *
                                             (world_origin_target - world_origin_base),
                                         base_jacobian, update_kinematics);

  RigidBodyDynamics::CalcPointJacobian6D(dynamic_model_.rbdl_model_, Q, target_body_id,
                                         Eigen::Vector3d::Zero(), target_jacobian,
                                         update_kinematics);

  Eigen::MatrixXd R(6, 6);
  R.setZero();
  R.block(0, 0, 3, 3) = world_R_base.inverse();
  R.block(3, 3, 3, 3) = world_R_base.inverse();

  jacobian = R * (target_jacobian - base_jacobian);
}

void KangarooPhysicsSimulator::computeInverseGeneralizedInertia(Eigen::MatrixXd &inv_generalized_inertia,
                                                                size_t fixed_links_size)
{
  Eigen::VectorXd generalized_effort = dynamic_model_state_.generalized_effort_;
  Eigen::VectorXd generalized_vel = dynamic_model_state_.generalized_velocity_;
  Eigen::MatrixXd rel_jacobian_left, rel_jacobian_right, relative_jac;
  relative_jac.resize(6, dynamic_model_.getDOFNumber());
  rel_jacobian_left.resize(2, dynamic_model_.getDOFNumber());
  rel_jacobian_right.resize(2, dynamic_model_.getDOFNumber());
  generalized_effort(dynamic_model_.getJointIndex("leg_left_knee_joint") + 6) = 0.0;
  generalized_effort(dynamic_model_.getJointIndex("leg_left_femur_joint") + 6) = 0.0;
  generalized_effort(dynamic_model_.getJointIndex("leg_right_knee_joint") + 6) = 0.0;
  generalized_effort(dynamic_model_.getJointIndex("leg_right_femur_joint") + 6) = 0.0;

  // Formulating the relative jacobian
  const unsigned int a_left_frame = dynamic_model_.rbdl_model_.GetBodyId("leg_left_length_link");
  const unsigned int u_left_frame =
      dynamic_model_.rbdl_model_.GetBodyId("leg_left_knee_lower_bearing");
  const unsigned int a_right_frame =
      dynamic_model_.rbdl_model_.GetBodyId("leg_right_length_link");
  const unsigned int u_right_frame =
      dynamic_model_.rbdl_model_.GetBodyId("leg_right_knee_lower_bearing");
  this->computeRelativeJacobian(dynamic_model_state_.generalized_coordinates_,
                                a_left_frame, u_left_frame, relative_jac, false);
  rel_jacobian_left.block(0, 0, 1, dynamic_model_.getDOFNumber()) =
      relative_jac.block(3, 0, 1, dynamic_model_.getDOFNumber());
  rel_jacobian_left.block(1, 0, 1, dynamic_model_.getDOFNumber()) =
      relative_jac.block(5, 0, 1, dynamic_model_.getDOFNumber());

  this->computeRelativeJacobian(dynamic_model_state_.generalized_coordinates_,
                                a_right_frame, u_right_frame, relative_jac, false);
  rel_jacobian_right.block(0, 0, 1, dynamic_model_.getDOFNumber()) =
      relative_jac.block(3, 0, 1, dynamic_model_.getDOFNumber());
  rel_jacobian_right.block(1, 0, 1, dynamic_model_.getDOFNumber()) =
      relative_jac.block(5, 0, 1, dynamic_model_.getDOFNumber());

  if (fixed_links_size == 0)
  {
    // -----
    // inversion
    /*
     * matrix MJ:
     *   | M      -J_R_l^T -J_R_r^T|
     *   | -J_R_l    0        0    |
     *   | -J_R_r    0        0    |
     *
     *  M - The generalized inertia matrix
     *  J_R_l - The relative jacobian of the kangaroo left leg
     *  J_R_r - The relative jacobian of the kangaroo right leg
     */
    Eigen::MatrixXd MJ;
    MJ.resize(
        dynamic_model_.getDOFNumber() + rel_jacobian_left.rows() + rel_jacobian_right.rows(),
        dynamic_model_.getDOFNumber() + rel_jacobian_left.rows() + rel_jacobian_right.rows());

    MJ.topLeftCorner(dynamic_model_.getDOFNumber(), dynamic_model_.rbdl_model_.dof_count) =
        generalized_inertia_;
    MJ.bottomRightCorner(rel_jacobian_left.rows() + rel_jacobian_right.rows(),
                         rel_jacobian_left.rows() + rel_jacobian_right.rows())
        .setZero();

    MJ.block(dynamic_model_.getDOFNumber(), 0, rel_jacobian_left.rows(),
             dynamic_model_.getDOFNumber()) = -rel_jacobian_left;
    MJ.block(dynamic_model_.getDOFNumber() + rel_jacobian_left.rows(), 0,
             rel_jacobian_right.rows(), dynamic_model_.getDOFNumber()) = -rel_jacobian_right;
    MJ.block(0, dynamic_model_.rbdl_model_.dof_count, dynamic_model_.getDOFNumber(),
             rel_jacobian_left.rows()) = -rel_jacobian_left.transpose();
    MJ.block(0, dynamic_model_.rbdl_model_.dof_count + rel_jacobian_left.rows(),
             dynamic_model_.getDOFNumber(), rel_jacobian_right.rows()) =
        -rel_jacobian_right.transpose();
    inv_generalized_inertia = MJ.inverse().topLeftCorner(
        dynamic_model_.getDOFNumber(), dynamic_model_.rbdl_model_.dof_count);
  }
  else
  {
    // This is the case of the fixed base
    // This part need to be done to compute the inertia matrix and the inverted inertia
    // matrix that is needed later for the additional velocity components computation, as
    // the inertia matrix is different if the fixed links are present. If not, it is same
    // as the normal inertia matrix.
    // -----
    // Jacobian
    std::size_t jacobian_size = 0;

    for (std::size_t i = 0; i < simulation_parameters_.fixed_links_.size(); ++i)
    {
      jacobian_size += simulation_parameters_.fixed_links_[i].getJacobianRowsNum();
    }

    Eigen::MatrixXd jacobian;
    Eigen::MatrixXd jacobian_part;

    jacobian.resize(jacobian_size, dynamic_model_.getDOFNumber());

    std::size_t jacobian_index = 0;
    for (const auto &fixed_link : simulation_parameters_.fixed_links_)
    {
      dynamic_model_.getTagJacobian(fixed_link, jacobian_part);
      jacobian.middleRows(jacobian_index, jacobian_part.rows()) = jacobian_part;
      jacobian_index += jacobian_part.rows();
    }

    pal::math_utils::QRFactorization jacobian_qr(jacobian);
    jacobian_qr.reconstructLinearlyIndependent(jacobian, 1e-10);
    // -----


    // -----
    // inversion
    /*
     * matrix MJ:
     *   | M      -J_R_l^T -J_R_r^T  -Jf^T |
     *   | -J_R_l    0        0        0   |
     *   | -J_R_r    0        0        0   |
     *   | -Jf       0        0        0   |
     *
     *  M - The generalized inertia matrix
     *  J_R_l - The relative jacobian of the kangaroo left leg
     *  J_R_r - The relative jacobian of the kangaroo right leg
     *  Jf - The jacobian of the fixed links
     */
    const std::size_t kangaroo_jacobian_size =
        rel_jacobian_right.rows() + rel_jacobian_left.rows();
    Eigen::MatrixXd MJ;
    MJ.resize(dynamic_model_.getDOFNumber() + jacobian.rows() + kangaroo_jacobian_size,
              dynamic_model_.getDOFNumber() + jacobian.rows() + kangaroo_jacobian_size);

    MJ.topLeftCorner(dynamic_model_.getDOFNumber(), dynamic_model_.rbdl_model_.dof_count) =
        generalized_inertia_;
    MJ.bottomRightCorner(kangaroo_jacobian_size + jacobian.rows(),
                         kangaroo_jacobian_size + jacobian.rows())
        .setZero();

    MJ.block(dynamic_model_.getDOFNumber(), 0, rel_jacobian_left.rows(),
             dynamic_model_.getDOFNumber()) = -rel_jacobian_left;
    MJ.block(dynamic_model_.getDOFNumber() + rel_jacobian_left.rows(), 0,
             rel_jacobian_right.rows(), dynamic_model_.getDOFNumber()) = -rel_jacobian_right;

    MJ.block(dynamic_model_.getDOFNumber() + kangaroo_jacobian_size, 0, jacobian.rows(),
             dynamic_model_.getDOFNumber()) = -jacobian;

    MJ.block(0, dynamic_model_.rbdl_model_.dof_count, dynamic_model_.getDOFNumber(),
             rel_jacobian_left.rows()) = -rel_jacobian_left.transpose();
    MJ.block(0, dynamic_model_.rbdl_model_.dof_count + rel_jacobian_left.rows(),
             dynamic_model_.getDOFNumber(), rel_jacobian_right.rows()) =
        -rel_jacobian_right.transpose();
    MJ.block(0, dynamic_model_.rbdl_model_.dof_count + kangaroo_jacobian_size,
             dynamic_model_.getDOFNumber(), jacobian.rows()) = -jacobian.transpose();

    inv_generalized_inertia = MJ.inverse().topLeftCorner(
        dynamic_model_.getDOFNumber(), dynamic_model_.rbdl_model_.dof_count);
  }
}
}
PLUGINLIB_EXPORT_CLASS(pal_physics_simulator::KangarooPhysicsSimulator,
                       pal_physics_simulator::PhysicsSimulatorBase);
