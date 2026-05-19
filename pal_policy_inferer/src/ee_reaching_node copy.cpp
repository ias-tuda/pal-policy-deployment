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
// pose_command: x y z  qw qx qy qz
static std::vector<double> obs_pose_left  = {-0.55, 0.0, 0.18, 1.0, 0.0, 0.0, 0.0};
static std::vector<double> obs_pose_right = { 0.55, 0.0, 0.18, 1.0, 0.0, 0.0, 0.0};

static bool received_pose = false;

// ONNX
static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ee_reach");
static std::unique_ptr<Ort::Session> sess;
static Ort::MemoryInfo mem =
    Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
static std::vector<const char*> in_names, out_names;

static ros::Publisher pub_cmd;
static tf::TransformListener* tf_listener = nullptr;

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

static void logEEError(const std::string& ee_frame,
                       const std::vector<double>& target,
                       const char* side) {
    if (!tf_listener) return;
    try {
        tf::StampedTransform tf;
        tf_listener->lookupTransform("imu_link", ee_frame, ros::Time(0), tf);
        const tf::Vector3& pos = tf.getOrigin();
        const tf::Quaternion ee_q = tf.getRotation();

        const double pos_err = posDist(target, pos);
        const double ori_err_deg = target.size() >= 7 ? quatErrDeg(target, ee_q) : 999.0;

        const bool pos_ok = pos_err < kPosReachedThreshM;
        const bool ori_ok = ori_err_deg < kOriReachedThreshDeg;
        const bool reached = pos_ok && ori_ok;
        const char* status = reached ? "REACHED" : "reaching";

        ROS_INFO_THROTTLE(1.0,
            "[ee_reach] %s  target=[%.3f %.3f %.3f]  ee=[%.3f %.3f %.3f]  "
            "pos_err=%.3f m (thr=%.3f)  ori_err=%.1f deg (thr=%.1f)  "
            "[%s pos:%s ori:%s]",
            side,
            target[0], target[1], target[2],
            pos.x(), pos.y(), pos.z(),
            pos_err, kPosReachedThreshM,
            ori_err_deg, kOriReachedThreshDeg,
            status,
            pos_ok ? "ok" : "bad",
            ori_ok ? "ok" : "bad");
    } catch (const tf::TransformException&) {
        ROS_WARN_THROTTLE(5.0, "[ee_reach] No TF imu_link -> %s yet", ee_frame.c_str());
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

    std::vector<double> targets(N);
    {
        std::lock_guard<std::mutex> lk(obs_mutex);
        for (int i = 0; i < N; ++i) {
            obs_actions[i] = y[i];                              // raw → next obs
            targets[i]     = action_scale[i] * y[i] + default_pos[i];
        }
    }

    sensor_msgs::JointState cmd;
    cmd.header.stamp = ros::Time::now();
    cmd.name         = joint_names;
    cmd.position     = targets;
    cmd.velocity.assign(N, 0.0);
    cmd.effort.assign(N, 0.0);
    pub_cmd.publish(cmd);

    // EE distance to target (1 Hz, in imu_link frame)
    std::vector<double> tgt_l, tgt_r;
    {
        std::lock_guard<std::mutex> lk(obs_mutex);
        tgt_l = obs_pose_left;
        tgt_r = obs_pose_right;
    }
    logEEError("arm_left_7_link",  tgt_l, "LEFT ");
    logEEError("arm_right_7_link", tgt_r, "RIGHT");
}

// ============================================================
//  /joint_states
// ============================================================
static void jsCB(const sensor_msgs::JointState::ConstPtr& msg) {
    std::lock_guard<std::mutex> lk(obs_mutex);
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
    ROS_INFO_THROTTLE(2.0, "[ee_reach] %s  frame=%s  "
        "xyz=[%.3f %.3f %.3f]  qwxyz=[%.3f %.3f %.3f %.3f]",
        tag, msg->header.frame_id.c_str(),
        p.x, p.y, p.z, q.w, q.x, q.y, q.z);
    {
        std::lock_guard<std::mutex> lk(obs_mutex);
        slot = {p.x, p.y, p.z, q.w, q.x, q.y, q.z};
    }
    received_pose = true;
}

static void poseLeftCB(const geometry_msgs::PoseStamped::ConstPtr& m) {
    poseCB(m, obs_pose_left, "left_pose ");
}
static void poseRightCB(const geometry_msgs::PoseStamped::ConstPtr& m) {
    poseCB(m, obs_pose_right, "right_pose");
}

// ============================================================
//  MAIN
// ============================================================
int main(int argc, char** argv) {
    ros::init(argc, argv, "ee_reaching_node");
    ros::NodeHandle nh("~");

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
    pub_cmd = nh.advertise<sensor_msgs::JointState>(
        "/subscriber_controller/desired_state", 1);

    tf::TransformListener tfl;
    tf_listener = &tfl;

    ros::Subscriber sub_js    = nh.subscribe("/joint_states",        1, jsCB);
    ros::Subscriber sub_left  = nh.subscribe("/reaching/left_pose",  1, poseLeftCB);
    ros::Subscriber sub_right = nh.subscribe("/reaching/right_pose", 1, poseRightCB);

    ros::Timer timer = nh.createTimer(ros::Duration(0.02), timerCB);

    ROS_INFO("[ee_reach] Ready. Publish pose to /reaching/left_pose (frame_id: imu_link)");
    ros::spin();
    return 0;
}
