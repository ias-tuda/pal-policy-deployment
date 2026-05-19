/**
 * ee_reaching_node
 *
 * Observation (56 values, no history):
 *   [ 0..13]  joint_pos_rel  = joint_pos - default_joint_pos   (/joint_states)
 *   [14..27]  joint_vel                                         (/joint_states)
 *   [28..41]  actions        = raw ONNX output from last step   (intern)
 *   [42..48]  pose_command_left   x y z qw qx qy qz            (/reaching/left_pose)
 *   [49..55]  pose_command_right  x y z qw qx qy qz            (/reaching/right_pose)
 *
 * Action:
 *   joint_target[i] = action_scale[i] * y[i] + default_joint_pos[i]
 *   -> /subscriber_controller/desired_state  at 50 Hz
 *
 * Start:
 *   rosrun pal_policy_inferer ee_reaching_node \
 *     _model_path:=/path/to/model.onnx
 */

#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Float64MultiArray.h>
#include <onnxruntime_cxx_api.h>
#include <tf/transform_listener.h>
#include <vector>
#include <string>
#include <map>
#include <mutex>
#include <sstream>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <limits>
#include <iostream>
#include <atomic>
#include <thread>

// ============================================================
//  Protobuf mini-parser for ONNX metadata
//  Same approach as sim_policy.py / read_onnx_meta.py
//  — avoids ORT metadata API which has issues with some models
// ============================================================
static uint64_t pbVarint(const uint8_t* d, size_t& pos, size_t sz) {
    uint64_t r = 0; int sh = 0;
    while (pos < sz) {
        uint8_t b = d[pos++];
        r |= (uint64_t)(b & 0x7F) << sh;
        if (!(b & 0x80)) break;
        sh += 7;
    }
    return r;
}

static void pbEntry(const uint8_t* d, size_t sz, std::string& key, std::string& val) {
    size_t pos = 0;
    while (pos < sz) {
        uint64_t tag = pbVarint(d, pos, sz);
        int fn = tag >> 3, wt = tag & 0x7;
        if (wt != 2) break;
        uint64_t len = pbVarint(d, pos, sz);
        std::string s(reinterpret_cast<const char*>(d + pos), len);
        pos += len;
        if (fn == 1) key = s;
        else if (fn == 2) val = s;
    }
}

static std::map<std::string, std::string> onnxMeta(const std::string& path) {
    std::map<std::string, std::string> m;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return m;
    fseek(f, 0, SEEK_END);
    size_t sz = static_cast<size_t>(ftell(f));
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> data(sz);
    fread(data.data(), 1, sz, f);
    fclose(f);

    size_t pos = 0;
    while (pos < sz) {
        size_t sp = pos;
        uint64_t tag = pbVarint(data.data(), pos, sz);
        int fn = tag >> 3, wt = tag & 0x7;
        if (wt == 2) {
            uint64_t len = pbVarint(data.data(), pos, sz);
            if (fn == 14) {  // metadata_props
                std::string k, v;
                pbEntry(data.data() + pos, len, k, v);
                if (!k.empty()) m[k] = v;
            }
            pos += len;
        } else if (wt == 0) { pbVarint(data.data(), pos, sz); }
        else if (wt == 1) { pos += 8; }
        else if (wt == 5) { pos += 4; }
        else break;
        if (pos <= sp) break;
    }
    return m;
}

// ============================================================
//  String helpers
// ============================================================
static std::vector<std::string> csv(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        while (!tok.empty() && tok.front() == ' ') tok.erase(tok.begin());
        while (!tok.empty() && tok.back()  == ' ') tok.pop_back();
        if (!tok.empty()) out.push_back(tok);
    }
    return out;
}

static std::vector<double> csvd(const std::string& s) {
    std::vector<double> out;
    for (auto& t : csv(s)) out.push_back(std::stod(t));
    return out;
}

// ============================================================
//  Policy state (all protected by obs_mutex)
// ============================================================
static std::mutex obs_mutex;

static std::vector<std::string> joint_names;
static std::vector<double>      default_pos;
static std::vector<double>      action_scale;
static int                      N = 14;           // number of arm joints

static std::vector<double> obs_joint_pos;          // relative to default
static std::vector<double> obs_joint_vel;
static std::vector<double> obs_actions;            // raw ONNX output of last step

static double              max_joint_vel = 2.0;    // rad/s velocity safety limit
static std::vector<double> last_cmd_pos;           // previous commanded positions

static bool                hold_default_on_start = false;
// pose_command: x y z  qw qx qy qz
static std::vector<double> obs_pose_left  = {-0.55, 0.0, 0.18, 1.0, 0.0, 0.0, 0.0};
static std::vector<double> obs_pose_right = { 0.55, 0.0, 0.18, 1.0, 0.0, 0.0, 0.0};

static bool received_pose = false;

// Sensor timestamps — used for policy_input/* headers
static ros::Time stamp_js         = ros::Time(0);
static ros::Time stamp_pose_left  = ros::Time(0);
static ros::Time stamp_pose_right = ros::Time(0);

// ONNX
static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ee_reach");
static std::unique_ptr<Ort::Session> sess;
static Ort::MemoryInfo mem =
    Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
static std::vector<const char*> in_names, out_names;

static ros::Publisher pub_cmd;
static ros::Publisher pub_in_joint_pos;
static ros::Publisher pub_in_joint_vel;
static ros::Publisher pub_in_actions;
static ros::Publisher pub_in_pose_left;
static ros::Publisher pub_in_pose_right;
static ros::Publisher pub_in_obs;
static ros::Publisher pub_out_raw;
static ros::Publisher pub_out_scaled;
static ros::Publisher pub_reach_error_left;
static ros::Publisher pub_reach_error_right;
static ros::Publisher pub_reach_distance_left;
static ros::Publisher pub_reach_distance_right;
static ros::Publisher pub_reach_ori_error_left;
static ros::Publisher pub_reach_ori_error_right;
static ros::Publisher pub_reach_reached_left;
static ros::Publisher pub_reach_reached_right;
static ros::Publisher pub_reach_ee_pose_left;
static ros::Publisher pub_reach_ee_pose_right;
static tf::TransformListener* tf_listener = nullptr;

static std::string base_frame = "imu_link";
static std::string left_ee_frame = "arm_left_7_link";
static std::string right_ee_frame = "arm_right_7_link";

static constexpr double kPosReachedThreshM = 0.05;
static constexpr double kOriReachedThreshDeg = 15.0;

static double posDist(const std::vector<double>& target, const tf::Vector3& ee) {
    double dx = target[0] - ee.x();
    double dy = target[1] - ee.y();
    double dz = target[2] - ee.z();
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

static double quatErrDeg(const std::vector<double>& target, const tf::Quaternion& ee_quat) {
    tf::Quaternion q_target(target[4], target[5], target[6], target[3]); // x y z w
    tf::Quaternion q_ee = ee_quat;
    q_target.normalize();
    q_ee.normalize();

    double dot = q_target.x() * q_ee.x() +
                 q_target.y() * q_ee.y() +
                 q_target.z() * q_ee.z() +
                 q_target.w() * q_ee.w();
    dot = std::fabs(dot);                    // q and -q are same orientation
    if (dot > 1.0) dot = 1.0;

    const double angle_rad = 2.0 * std::acos(dot);
    return angle_rad * 180.0 / M_PI;
}

static void publishEEError(const std::string& ee_frame,
                           const std::vector<double>& target,
                           const char* side,
                           const ros::Publisher& pub_error,
                           const ros::Publisher& pub_distance,
                           const ros::Publisher& pub_ori_error,
                           const ros::Publisher& pub_reached,
                           const ros::Publisher& pub_ee_pose) {
    if (!tf_listener) return;
    try {
        tf::StampedTransform tf;
        tf_listener->lookupTransform(base_frame, ee_frame, ros::Time(0), tf);
        const tf::Vector3& pos = tf.getOrigin();
        const tf::Quaternion ee_q = tf.getRotation();

        const double dx = target[0] - pos.x();
        const double dy = target[1] - pos.y();
        const double dz = target[2] - pos.z();
        const double pos_err = posDist(target, pos);
        const double ori_err_deg = target.size() >= 7 ? quatErrDeg(target, ee_q) : 999.0;

        const bool pos_ok = pos_err < kPosReachedThreshM;
        const bool ori_ok = ori_err_deg < kOriReachedThreshDeg;
        const bool reached = pos_ok && ori_ok;
        const char* status = reached ? "REACHED" : "reaching";

        std_msgs::Float64MultiArray err_msg;
        err_msg.data = {dx, dy, dz, pos_err, ori_err_deg, reached ? 1.0 : 0.0};
        pub_error.publish(err_msg);

        std_msgs::Float64 distance_msg;
        distance_msg.data = pos_err;
        pub_distance.publish(distance_msg);

        std_msgs::Float64 ori_msg;
        ori_msg.data = ori_err_deg;
        pub_ori_error.publish(ori_msg);

        std_msgs::Bool reached_msg;
        reached_msg.data = reached;
        pub_reached.publish(reached_msg);

        geometry_msgs::PoseStamped ee_pose;
        ee_pose.header.stamp = ros::Time::now();
        ee_pose.header.frame_id = base_frame;
        ee_pose.pose.position.x = pos.x();
        ee_pose.pose.position.y = pos.y();
        ee_pose.pose.position.z = pos.z();
        ee_pose.pose.orientation.x = ee_q.x();
        ee_pose.pose.orientation.y = ee_q.y();
        ee_pose.pose.orientation.z = ee_q.z();
        ee_pose.pose.orientation.w = ee_q.w();
        pub_ee_pose.publish(ee_pose);

        ROS_INFO_THROTTLE(1.0,
            "[ee_reach] %s  frame=%s target=[%.3f %.3f %.3f]  ee=[%.3f %.3f %.3f]  "
            "pos_err=%.3f m (thr=%.3f)  ori_err=%.1f deg (thr=%.1f)  "
            "[%s pos:%s ori:%s]",
            side,
            base_frame.c_str(),
            target[0], target[1], target[2],
            pos.x(), pos.y(), pos.z(),
            pos_err, kPosReachedThreshM,
            ori_err_deg, kOriReachedThreshDeg,
            status,
            pos_ok ? "ok" : "bad",
            ori_ok ? "ok" : "bad");
    } catch (const tf::TransformException&) {
        ROS_WARN_THROTTLE(5.0, "[ee_reach] No TF %s -> %s yet",
                          base_frame.c_str(), ee_frame.c_str());
    }
}

// ============================================================
//  Build obs vector: [joint_pos|joint_vel|actions|pose_L|pose_R]
// ============================================================
static std::vector<float> buildObs() {
    std::vector<float> o;
    o.reserve(56);
    for (double v : obs_joint_pos)   o.push_back(static_cast<float>(v));
    for (double v : obs_joint_vel)   o.push_back(static_cast<float>(v));
    for (double v : obs_actions)     o.push_back(static_cast<float>(v));
    for (double v : obs_pose_left)   o.push_back(static_cast<float>(v));
    for (double v : obs_pose_right)  o.push_back(static_cast<float>(v));
    return o;
}

// ============================================================
//  50 Hz timer — run ONNX inference
// ============================================================
static void timerCB(const ros::TimerEvent&) {
    if (!received_pose) {
        ROS_WARN_THROTTLE(2.0, "[ee_reach] Waiting for pose on "
                               "/reaching/left_pose or /reaching/right_pose");
        return;
    }

    std::vector<float> obs;
    {
        std::lock_guard<std::mutex> lk(obs_mutex);
        obs = buildObs();
    }

    std::vector<int64_t> shape = {1, 56};
    Ort::Value in_t = Ort::Value::CreateTensor<float>(
        mem, obs.data(), obs.size(), shape.data(), shape.size());

    auto out = sess->Run(Ort::RunOptions{nullptr},
                         in_names.data(), &in_t, 1, out_names.data(), 1);
    const float* y = out.front().GetTensorData<float>();

    static bool first_inference = true;

    std::vector<double> targets(N);
    {
        std::lock_guard<std::mutex> lk(obs_mutex);
        bool init_from_default = last_cmd_pos.empty();
        if (init_from_default) last_cmd_pos = default_pos;
        const double max_delta = max_joint_vel * 0.02;  // dt = 1/50 Hz
        for (int i = 0; i < N; ++i) {
            obs_actions[i] = y[i];                              // raw → next obs
            double raw = action_scale[i] * y[i] + default_pos[i];
            double delta = raw - last_cmd_pos[i];
            delta = std::max(-max_delta, std::min(max_delta, delta));
            targets[i] = last_cmd_pos[i] + delta;
        }
        last_cmd_pos = targets;

        if (first_inference) {
            first_inference = false;
            ROS_WARN("[ee_reach] === FIRST INFERENCE STEP ===");
            ROS_WARN("[ee_reach]   last_cmd_pos init from: %s",
                     init_from_default ? "default_pos" : "previous cmd");
            ROS_WARN("[ee_reach]   obs[0..13]  joint_pos_rel:");
            for (int i = 0; i < N; ++i)
                ROS_WARN("[ee_reach]     [%2d] %-28s  pos_rel=%+.4f  vel=%+.4f",
                         i, joint_names[i].c_str(), obs_joint_pos[i], obs_joint_vel[i]);
            ROS_WARN("[ee_reach]   obs[42..48] pose_left:  xyz=[%.3f %.3f %.3f] qwxyz=[%.3f %.3f %.3f %.3f]",
                     obs_pose_left[0], obs_pose_left[1], obs_pose_left[2],
                     obs_pose_left[3], obs_pose_left[4], obs_pose_left[5], obs_pose_left[6]);
            ROS_WARN("[ee_reach]   obs[49..55] pose_right: xyz=[%.3f %.3f %.3f] qwxyz=[%.3f %.3f %.3f %.3f]",
                     obs_pose_right[0], obs_pose_right[1], obs_pose_right[2],
                     obs_pose_right[3], obs_pose_right[4], obs_pose_right[5], obs_pose_right[6]);
            ROS_WARN("[ee_reach]   raw ONNX output  →  scaled target  (max_delta=%.4f):", max_delta);
            for (int i = 0; i < N; ++i) {
                double raw = action_scale[i] * y[i] + default_pos[i];
                double delta = targets[i] - default_pos[i];
                ROS_WARN("[ee_reach]     [%2d] %-28s  y=%.4f  scaled=%.4f  target=%.4f  delta_from_def=%+.4f",
                         i, joint_names[i].c_str(), y[i], raw, targets[i], delta);
            }
        }
    }

    ros::Time now = ros::Time::now();

    sensor_msgs::JointState cmd;
    cmd.header.stamp = now;
    cmd.name         = joint_names;
    cmd.position     = targets;
    cmd.velocity.assign(N, 0.0);
    cmd.effort.assign(N, 0.0);
    pub_cmd.publish(cmd);

    // ── policy_input/* ──────────────────────────────────────────
    {
        std::lock_guard<std::mutex> lk(obs_mutex);

        sensor_msgs::JointState js_pos, js_vel, js_act;
        js_pos.header.stamp = js_vel.header.stamp = js_act.header.stamp = stamp_js;
        js_pos.name = js_vel.name = js_act.name = joint_names;
        js_pos.position  = obs_joint_pos;
        js_vel.position  = obs_joint_vel;
        js_act.position  = obs_actions;
        pub_in_joint_pos.publish(js_pos);
        pub_in_joint_vel.publish(js_vel);
        pub_in_actions.publish(js_act);

        geometry_msgs::PoseStamped pl, pr;
        pl.header.stamp = stamp_pose_left;
        pr.header.stamp = stamp_pose_right;
        pl.header.frame_id = pr.header.frame_id = "imu_link";
        pl.pose.position.x = obs_pose_left[0];
        pl.pose.position.y = obs_pose_left[1];
        pl.pose.position.z = obs_pose_left[2];
        pl.pose.orientation.w = obs_pose_left[3];
        pl.pose.orientation.x = obs_pose_left[4];
        pl.pose.orientation.y = obs_pose_left[5];
        pl.pose.orientation.z = obs_pose_left[6];
        pr.pose.position.x = obs_pose_right[0];
        pr.pose.position.y = obs_pose_right[1];
        pr.pose.position.z = obs_pose_right[2];
        pr.pose.orientation.w = obs_pose_right[3];
        pr.pose.orientation.x = obs_pose_right[4];
        pr.pose.orientation.y = obs_pose_right[5];
        pr.pose.orientation.z = obs_pose_right[6];
        pub_in_pose_left.publish(pl);
        pub_in_pose_right.publish(pr);

        std_msgs::Float64MultiArray obs_msg;
        for (float v : obs) obs_msg.data.push_back(static_cast<double>(v));
        pub_in_obs.publish(obs_msg);
    }

    // ── policy_output/* ─────────────────────────────────────────
    {
        std::vector<double> raw_scaled(N);
        for (int i = 0; i < N; ++i)
            raw_scaled[i] = action_scale[i] * y[i] + default_pos[i];

        sensor_msgs::JointState js_raw, js_scaled;
        js_raw.header.stamp = js_scaled.header.stamp = now;
        js_raw.name = js_scaled.name = joint_names;
        js_raw.position.assign(y, y + N);   // raw network output
        js_scaled.position = raw_scaled;    // scaled (before vel clamp)
        pub_out_raw.publish(js_raw);
        pub_out_scaled.publish(js_scaled);
    }

    // EE distance to target in base_frame.
    std::vector<double> tgt_l, tgt_r;
    {
        std::lock_guard<std::mutex> lk(obs_mutex);
        tgt_l = obs_pose_left;
        tgt_r = obs_pose_right;
    }
    publishEEError(left_ee_frame,  tgt_l, "LEFT ",
                   pub_reach_error_left, pub_reach_distance_left,
                   pub_reach_ori_error_left, pub_reach_reached_left,
                   pub_reach_ee_pose_left);
    publishEEError(right_ee_frame, tgt_r, "RIGHT",
                   pub_reach_error_right, pub_reach_distance_right,
                   pub_reach_ori_error_right, pub_reach_reached_right,
                   pub_reach_ee_pose_right);
}

// ============================================================
//  /joint_states
// ============================================================
static void jsCB(const sensor_msgs::JointState::ConstPtr& msg) {
    std::lock_guard<std::mutex> lk(obs_mutex);
    stamp_js = msg->header.stamp;
    for (int k = 0; k < N; ++k)
        for (size_t i = 0; i < msg->name.size(); ++i)
            if (msg->name[i] == joint_names[k]) {
                obs_joint_pos[k] = msg->position[i] - default_pos[k];
                obs_joint_vel[k] = msg->velocity[i];
                break;
            }
}

// ============================================================
//  Pose callbacks  (frame_id: imu_link, quat order: w x y z)
// ============================================================
static void poseCB(const geometry_msgs::PoseStamped::ConstPtr& msg,
                   std::vector<double>& slot, const char* tag) {
    const auto& p = msg->pose.position;
    const auto& q = msg->pose.orientation;
    bool was_first = !received_pose;
    ROS_INFO_THROTTLE(2.0, "[ee_reach] %s  frame=%s  "
        "xyz=[%.3f %.3f %.3f]  qwxyz=[%.3f %.3f %.3f %.3f]",
        tag, msg->header.frame_id.c_str(),
        p.x, p.y, p.z, q.w, q.x, q.y, q.z);
    {
        std::lock_guard<std::mutex> lk(obs_mutex);
        slot = {p.x, p.y, p.z, q.w, q.x, q.y, q.z};
    }
    received_pose = true;
    if (was_first)
        ROS_WARN("[ee_reach] FIRST POSE received (%s) — inference starts NOW. "
                 "joint_pos_rel will be used as-is from current /joint_states.", tag);
}

static void poseLeftCB(const geometry_msgs::PoseStamped::ConstPtr& m) {
    stamp_pose_left = m->header.stamp;
    poseCB(m, obs_pose_left, "left_pose ");
}
static void poseRightCB(const geometry_msgs::PoseStamped::ConstPtr& m) {
    stamp_pose_right = m->header.stamp;
    poseCB(m, obs_pose_right, "right_pose");
}

// ============================================================
//  MAIN
// ============================================================
int main(int argc, char** argv) {
    ros::init(argc, argv, "ee_reaching_node");
    ros::NodeHandle nh("~");

    nh.param<double>("max_joint_vel", max_joint_vel, 2.0);
    nh.param<bool>("hold_default_on_start", hold_default_on_start, false);
    nh.param<std::string>("base_frame", base_frame, "imu_link");
    nh.param<std::string>("left_ee_frame", left_ee_frame, "arm_left_7_link");
    nh.param<std::string>("right_ee_frame", right_ee_frame, "arm_right_7_link");
    ROS_INFO("[ee_reach] max_joint_vel: %.2f rad/s", max_joint_vel);
    ROS_INFO("[ee_reach] reach error frames: base=%s left_ee=%s right_ee=%s",
             base_frame.c_str(), left_ee_frame.c_str(), right_ee_frame.c_str());

    std::string model_path;
    nh.param<std::string>("model_path", model_path,
        "/catkin_ws/src/pal_policy_deployer/pal_policy_deployer"
        "/models/reach/2026-05-01_10-16-23_arms_fixed_reaching"
        "/2026-05-01_10-16-23_arms_fixed_reaching.onnx");

    ROS_INFO("[ee_reach] model: %s", model_path.c_str());

    // --- Parse metadata without ORT API ---
    auto meta = onnxMeta(model_path);
    if (meta.empty()) {
        ROS_FATAL("[ee_reach] Cannot read ONNX metadata from %s", model_path.c_str());
        return 1;
    }
    for (const char* k : {"joint_names", "default_joint_pos", "action_scale"})
        if (!meta.count(k)) { ROS_FATAL("[ee_reach] Missing key: %s", k); return 1; }

    joint_names  = csv(meta["joint_names"]);
    default_pos  = csvd(meta["default_joint_pos"]);
    action_scale = csvd(meta["action_scale"]);
    N = static_cast<int>(joint_names.size());

    obs_joint_pos.assign(N, 0.0);
    obs_joint_vel.assign(N, 0.0);
    obs_actions.assign(N, 0.0);

    // --- Print summary ---
    ROS_INFO("=== ee_reaching_node ===");
    ROS_INFO("  run_path : %s", meta.count("run_path") ? meta["run_path"].c_str() : "?");
    ROS_INFO("  joints   : %d", N);
    ROS_INFO("  obs_dim  : 56  (14+14+14+7+7, no history)");
    ROS_INFO("  Joint  |  default_pos  |  action_scale");
    for (int i = 0; i < N; ++i)
        ROS_INFO("  [%2d] %-28s  %+.4f  %.4f",
            i, joint_names[i].c_str(),
            i < (int)default_pos.size()  ? default_pos[i]  : 0.0,
            i < (int)action_scale.size() ? action_scale[i] : 0.0);

    // --- ONNX session ---
    ROS_INFO("[ee_reach] Loading ONNX model...");
    try {
        sess.reset(new Ort::Session(env, model_path.c_str(), Ort::SessionOptions{}));
    } catch (const std::exception& e) {
        ROS_FATAL("[ee_reach] ONNX load failed: %s", e.what());
        return 1;
    }
    ROS_INFO("[ee_reach] ONNX model loaded.");

    {
        Ort::AllocatorWithDefaultOptions alloc;
        in_names.push_back(sess->GetInputNameAllocated(0,  alloc).release());
        out_names.push_back(sess->GetOutputNameAllocated(0, alloc).release());
    }

    // --- ROS ---
    pub_cmd          = nh.advertise<sensor_msgs::JointState>("/subscriber_controller/desired_state", 1);
    pub_in_joint_pos = nh.advertise<sensor_msgs::JointState>("/policy_input/joint_pos",   1);
    pub_in_joint_vel = nh.advertise<sensor_msgs::JointState>("/policy_input/joint_vel",   1);
    pub_in_actions   = nh.advertise<sensor_msgs::JointState>("/policy_input/actions",     1);
    pub_in_pose_left = nh.advertise<geometry_msgs::PoseStamped>("/policy_input/pose_left",  1);
    pub_in_pose_right= nh.advertise<geometry_msgs::PoseStamped>("/policy_input/pose_right", 1);
    pub_in_obs       = nh.advertise<std_msgs::Float64MultiArray>("/policy_input/obs_vector", 1);
    pub_out_raw      = nh.advertise<sensor_msgs::JointState>("/policy_output/actions_raw",    1);
    pub_out_scaled   = nh.advertise<sensor_msgs::JointState>("/policy_output/actions_scaled", 1);
    pub_reach_error_left       = nh.advertise<std_msgs::Float64MultiArray>("/reaching/error_left", 1);
    pub_reach_error_right      = nh.advertise<std_msgs::Float64MultiArray>("/reaching/error_right", 1);
    pub_reach_distance_left    = nh.advertise<std_msgs::Float64>("/reaching/distance_left", 1);
    pub_reach_distance_right   = nh.advertise<std_msgs::Float64>("/reaching/distance_right", 1);
    pub_reach_ori_error_left   = nh.advertise<std_msgs::Float64>("/reaching/orientation_error_left", 1);
    pub_reach_ori_error_right  = nh.advertise<std_msgs::Float64>("/reaching/orientation_error_right", 1);
    pub_reach_reached_left     = nh.advertise<std_msgs::Bool>("/reaching/reached_left", 1);
    pub_reach_reached_right    = nh.advertise<std_msgs::Bool>("/reaching/reached_right", 1);
    pub_reach_ee_pose_left     = nh.advertise<geometry_msgs::PoseStamped>("/reaching/current_ee_pose_left", 1);
    pub_reach_ee_pose_right    = nh.advertise<geometry_msgs::PoseStamped>("/reaching/current_ee_pose_right", 1);

    tf::TransformListener tfl;
    tf_listener = &tfl;

    ros::Subscriber sub_js    = nh.subscribe("/joint_states",        1, jsCB);
    ros::Subscriber sub_left  = nh.subscribe("/reaching/left_pose",  1, poseLeftCB);
    ros::Subscriber sub_right = nh.subscribe("/reaching/right_pose", 1, poseRightCB);

    // ── Hold default_pos until user confirms ─────────────────────
    if (hold_default_on_start) {
        // Wait for first /joint_states
        ROS_INFO("[ee_reach] Waiting for first /joint_states...");
        ros::Rate wait_rate(50);
        while (ros::ok()) {
            ros::spinOnce();
            bool got = false;
            { std::lock_guard<std::mutex> lk(obs_mutex);
              for (int i = 0; i < N; ++i) if (obs_joint_pos[i] != 0.0) { got = true; break; } }
            if (got) break;
            wait_rate.sleep();
        }

        // Print actual vs default
        ROS_WARN("[ee_reach] === HOLDING DEFAULT_POS — actual vs default ===");
        ROS_WARN("[ee_reach]   %-28s  %10s  %10s  %10s", "Joint", "actual", "default", "error");
        {
            std::lock_guard<std::mutex> lk(obs_mutex);
            for (int i = 0; i < N; ++i) {
                double actual = obs_joint_pos[i] + default_pos[i];
                double err    = actual - default_pos[i];
                ROS_WARN("[ee_reach]   %-28s  %+10.4f  %+10.4f  %+10.4f%s",
                         joint_names[i].c_str(), actual, default_pos[i], err,
                         std::fabs(err) > 0.05 ? "  !!!" : "");
            }
        }

        // Publish default_pos at 50 Hz — impedance controller drives there smoothly
        ROS_WARN("[ee_reach] Publishing default_pos at 50 Hz (impedance drives robot there).");
        ROS_WARN("[ee_reach] Press ENTER when robot is in default_pos to start policy...");
        sensor_msgs::JointState cmd;
        cmd.name     = joint_names;
        cmd.position = default_pos;
        cmd.velocity.assign(N, 0.0);
        cmd.effort.assign(N, 0.0);

        ros::Rate hold_rate(50);
        std::atomic<bool> holding(true);
        int print_counter = 0;
        std::thread hold_thread([&]() {
            while (holding && ros::ok()) {
                cmd.header.stamp = ros::Time::now();
                pub_cmd.publish(cmd);
                ros::spinOnce();

                // Print actual vs default every 2 s (100 cycles at 50 Hz)
                if (++print_counter % 100 == 0) {
                    std::lock_guard<std::mutex> lk(obs_mutex);
                    bool all_ok = true;
                    for (int i = 0; i < N; ++i)
                        if (std::fabs(obs_joint_pos[i]) > 0.05) { all_ok = false; break; }
                    fprintf(stderr,
                        "\r\033[K[ee_reach] actual vs default (err):  ");
                    for (int i = 0; i < N; ++i) {
                        double err = obs_joint_pos[i]; // already relative to default
                        fprintf(stderr, "%s%+.3f%s ",
                            std::fabs(err) > 0.05 ? "\033[31m" : "\033[32m",
                            err,
                            "\033[0m");
                    }
                    fprintf(stderr, "%s\n",
                        all_ok ? "  \033[32m[OK]\033[0m" : "  \033[33m[moving...]\033[0m");
                    fflush(stderr);
                }
                hold_rate.sleep();
            }
        });

        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        holding = false;
        hold_thread.join();

        // Seed last_cmd_pos from actual position at this moment
        {
            std::lock_guard<std::mutex> lk(obs_mutex);
            last_cmd_pos.resize(N);
            for (int i = 0; i < N; ++i)
                last_cmd_pos[i] = obs_joint_pos[i] + default_pos[i];
        }
        ROS_WARN("[ee_reach] Starting policy. Publish pose to /reaching/left_pose (frame_id: imu_link)");
    } else {
        ROS_INFO("[ee_reach] Ready. Publish pose to /reaching/left_pose (frame_id: imu_link)");
    }

    ros::Timer timer = nh.createTimer(ros::Duration(0.02), timerCB);
    ros::spin();
    return 0;
}
