/**
 * ee_hold_default_node
 *
 * Observation (56 values, no history):
 *   [ 0..13]  joint_pos_rel  = joint_pos - default_joint_pos   (/joint_states)
 *   [14..27]  joint_vel                                         (/joint_states)
 *   [28..41]  actions        = raw ONNX output from last step   (intern)
 *   [42..48]  pose_command_left   x y z qw qx qy qz            (fixed zeros)
 *   [49..55]  pose_command_right  x y z qw qx qy qz            (fixed zeros)
 *
 * Like ee_reaching_node but starts immediately without waiting for a pose.
 * Designed for policies trained to hold the default joint configuration.
 *
 * Start:
 *   rosrun pal_policy_inferer pal_policy_inferer_ee_hold_default_node \
 *     _model_path:=/path/to/model.onnx \
 *     _max_joint_vel:=1.0
 */

#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Float64MultiArray.h>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <map>
#include <mutex>
#include <sstream>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <limits>

// ============================================================
//  Protobuf mini-parser for ONNX metadata
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
            if (fn == 14) {
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
//  Policy state
// ============================================================
static std::mutex obs_mutex;

static std::vector<std::string> joint_names;
static std::vector<double>      default_pos;
static std::vector<double>      action_scale;
static int                      N = 14;

static std::vector<double> obs_joint_pos;
static std::vector<double> obs_joint_abs_pos;
static std::vector<double> obs_joint_vel;
static std::vector<double> obs_actions;
static int                 obs_dim = 42;  // read from model at startup

static double              max_joint_vel = 2.0;
static bool                debug_print = true;
static double              debug_period = 0.5;
static bool                wait_for_joint_states = true;
static std::vector<double> last_cmd_pos;
static size_t              joint_state_count = 0;
static size_t              policy_tick_count = 0;
static bool                have_joint_states = false;
static bool                first_command_logged = false;
static ros::Time           last_debug_time = ros::Time(0);

static ros::Time stamp_js = ros::Time(0);

// ONNX
static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ee_hold");
static std::unique_ptr<Ort::Session> sess;
static Ort::MemoryInfo mem =
    Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
static std::vector<const char*> in_names, out_names;

// Publishers
static ros::Publisher pub_cmd;
static ros::Publisher pub_in_joint_pos;
static ros::Publisher pub_in_joint_vel;
static ros::Publisher pub_in_actions;
static ros::Publisher pub_in_obs;
static ros::Publisher pub_out_raw;
static ros::Publisher pub_out_scaled;

// ============================================================
//  Build obs vector
// ============================================================
static std::vector<float> buildObs() {
    std::vector<float> o;
    o.reserve(obs_dim);
    for (double v : obs_joint_pos) o.push_back(static_cast<float>(v));
    for (double v : obs_joint_vel) o.push_back(static_cast<float>(v));
    for (double v : obs_actions)   o.push_back(static_cast<float>(v));
    // pad with zeros if model expects more than 3*N
    while ((int)o.size() < obs_dim) o.push_back(0.0f);
    return o;
}

static void logVectorBlock(const std::string& label,
                           const std::vector<double>& values,
                           const std::vector<double>* reference = nullptr) {
    ROS_WARN_STREAM("[ee_hold_debug] " << label);
    for (int i = 0; i < N; ++i) {
        std::ostringstream line;
        line << "[ee_hold_debug]   [" << i << "] " << joint_names[i]
             << " " << values[i];
        if (reference && i < static_cast<int>(reference->size())) {
            line << " ref=" << (*reference)[i]
                 << " diff=" << (values[i] - (*reference)[i]);
        }
        ROS_WARN_STREAM(line.str());
    }
}

// ============================================================
//  50 Hz timer
// ============================================================
static void timerCB(const ros::TimerEvent&) {
    std::vector<float> obs;
    std::vector<double> abs_pos_snapshot;
    std::vector<double> rel_pos_snapshot;
    std::vector<double> vel_snapshot;
    std::vector<double> prev_cmd_snapshot;
    bool have_js_snapshot = false;
    size_t joint_state_count_snapshot = 0;
    ros::Time stamp_js_snapshot;
    {
        std::lock_guard<std::mutex> lk(obs_mutex);
        have_js_snapshot = have_joint_states;
        joint_state_count_snapshot = joint_state_count;
        stamp_js_snapshot = stamp_js;
        if (wait_for_joint_states && !have_joint_states) {
            ROS_WARN_THROTTLE(1.0, "[ee_hold] Waiting for first /joint_states before publishing commands.");
            return;
        }
        if (last_cmd_pos.empty()) {
            last_cmd_pos = have_joint_states ? obs_joint_abs_pos : default_pos;
            ROS_WARN_STREAM("[ee_hold] Initializing last_cmd_pos from "
                            << (have_joint_states ? "current /joint_states" : "default_pos"));
            if (debug_print) logVectorBlock("initial last_cmd_pos", last_cmd_pos, &default_pos);
        }
        obs = buildObs();
        abs_pos_snapshot = obs_joint_abs_pos;
        rel_pos_snapshot = obs_joint_pos;
        vel_snapshot = obs_joint_vel;
        prev_cmd_snapshot = last_cmd_pos;
    }

    std::vector<int64_t> shape = {1, static_cast<int64_t>(obs_dim)};
    Ort::Value in_t = Ort::Value::CreateTensor<float>(
        mem, obs.data(), obs.size(), shape.data(), shape.size());

    auto out = sess->Run(Ort::RunOptions{nullptr},
                         in_names.data(), &in_t, 1, out_names.data(), 1);
    const float* y = out.front().GetTensorData<float>();

    std::vector<double> targets(N);
    std::vector<double> raw_output(N);
    std::vector<double> raw_scaled(N);
    std::vector<double> applied_delta(N);
    {
        std::lock_guard<std::mutex> lk(obs_mutex);
        const double max_delta = max_joint_vel * 0.02;
        for (int i = 0; i < N; ++i) {
            obs_actions[i] = y[i];
            raw_output[i] = y[i];
            raw_scaled[i] = action_scale[i] * y[i] + default_pos[i];
            double raw = raw_scaled[i];
            double delta = raw - last_cmd_pos[i];
            delta = std::max(-max_delta, std::min(max_delta, delta));
            applied_delta[i] = delta;
            targets[i] = last_cmd_pos[i] + delta;
        }
        last_cmd_pos = targets;
        policy_tick_count++;
    }

    ros::Time now = ros::Time::now();

    if (debug_print &&
        (!first_command_logged ||
         last_debug_time.isZero() ||
         (now - last_debug_time).toSec() >= debug_period)) {
        first_command_logged = true;
        last_debug_time = now;
        ROS_WARN_STREAM("[ee_hold_debug] tick=" << policy_tick_count
                        << " have_js=" << have_js_snapshot
                        << " js_count=" << joint_state_count_snapshot
                        << " stamp_js=" << stamp_js_snapshot.toSec()
                        << " obs_dim=" << obs_dim
                        << " max_joint_vel=" << max_joint_vel);
        for (int i = 0; i < N; ++i) {
            const double rel = i < static_cast<int>(rel_pos_snapshot.size()) ? rel_pos_snapshot[i] : std::numeric_limits<double>::quiet_NaN();
            const double absq = i < static_cast<int>(abs_pos_snapshot.size()) ? abs_pos_snapshot[i] : std::numeric_limits<double>::quiet_NaN();
            const double vel = i < static_cast<int>(vel_snapshot.size()) ? vel_snapshot[i] : std::numeric_limits<double>::quiet_NaN();
            const double prev = i < static_cast<int>(prev_cmd_snapshot.size()) ? prev_cmd_snapshot[i] : std::numeric_limits<double>::quiet_NaN();
            ROS_WARN_STREAM("[ee_hold_debug]   " << joint_names[i]
                            << " q=" << absq
                            << " q_rel=" << rel
                            << " dq=" << vel
                            << " prev_cmd=" << prev
                            << " y_raw=" << raw_output[i]
                            << " scaled_target=" << raw_scaled[i]
                            << " applied_delta=" << applied_delta[i]
                            << " cmd=" << targets[i]
                            << " cmd_minus_q=" << (targets[i] - absq));
        }
    }

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
        js_pos.position = obs_joint_pos;
        js_vel.position = obs_joint_vel;
        js_act.position = obs_actions;
        pub_in_joint_pos.publish(js_pos);
        pub_in_joint_vel.publish(js_vel);
        pub_in_actions.publish(js_act);

        std_msgs::Float64MultiArray obs_msg;
        for (float v : obs) obs_msg.data.push_back(static_cast<double>(v));
        pub_in_obs.publish(obs_msg);
    }

    // ── policy_output/* ─────────────────────────────────────────
    {
        sensor_msgs::JointState js_raw, js_scaled;
        js_raw.header.stamp = js_scaled.header.stamp = now;
        js_raw.name = js_scaled.name = joint_names;
        js_raw.position = raw_output;
        js_scaled.position = raw_scaled;
        pub_out_raw.publish(js_raw);
        pub_out_scaled.publish(js_scaled);
    }
}

// ============================================================
//  /joint_states
// ============================================================
static void jsCB(const sensor_msgs::JointState::ConstPtr& msg) {
    std::lock_guard<std::mutex> lk(obs_mutex);
    stamp_js = msg->header.stamp;
    joint_state_count++;
    std::vector<std::string> missing;
    for (int k = 0; k < N; ++k) {
        bool found = false;
        for (size_t i = 0; i < msg->name.size(); ++i)
            if (msg->name[i] == joint_names[k]) {
                obs_joint_abs_pos[k] = msg->position[i];
                obs_joint_pos[k] = obs_joint_abs_pos[k] - default_pos[k];
                obs_joint_vel[k] = (i < msg->velocity.size()) ? msg->velocity[i] : 0.0;
                found = true;
                break;
            }
        if (!found) missing.push_back(joint_names[k]);
    }
    have_joint_states = missing.empty();
    if (!missing.empty()) {
        std::ostringstream oss;
        for (const auto& name : missing) oss << name << " ";
        ROS_WARN_THROTTLE(1.0, "[ee_hold] Missing joints in /joint_states: %s", oss.str().c_str());
    } else if (debug_print && joint_state_count <= 3) {
        ROS_WARN_STREAM("[ee_hold_debug] Received /joint_states #" << joint_state_count
                        << " stamp=" << stamp_js.toSec());
        logVectorBlock("current q from /joint_states", obs_joint_abs_pos, &default_pos);
        logVectorBlock("current dq from /joint_states", obs_joint_vel);
    }
}

// ============================================================
//  MAIN
// ============================================================
int main(int argc, char** argv) {
    ros::init(argc, argv, "ee_hold_default_node");
    ros::NodeHandle nh("~");

    nh.param<double>("max_joint_vel", max_joint_vel, 2.0);
    nh.param<bool>("debug_print", debug_print, true);
    nh.param<double>("debug_period", debug_period, 0.5);
    nh.param<bool>("wait_for_joint_states", wait_for_joint_states, true);
    ROS_INFO("[ee_hold] max_joint_vel: %.2f rad/s", max_joint_vel);
    ROS_INFO("[ee_hold] debug_print: %s debug_period: %.2f wait_for_joint_states: %s",
             debug_print ? "true" : "false",
             debug_period,
             wait_for_joint_states ? "true" : "false");

    std::string model_path;
    nh.param<std::string>("model_path", model_path, "");
    if (model_path.empty()) {
        ROS_FATAL("[ee_hold] _model_path required");
        return 1;
    }

    ROS_INFO("[ee_hold] model: %s", model_path.c_str());

    auto meta = onnxMeta(model_path);
    if (meta.empty()) {
        ROS_FATAL("[ee_hold] Cannot read ONNX metadata from %s", model_path.c_str());
        return 1;
    }
    for (const char* k : {"joint_names", "default_joint_pos", "action_scale"})
        if (!meta.count(k)) { ROS_FATAL("[ee_hold] Missing key: %s", k); return 1; }

    joint_names  = csv(meta["joint_names"]);
    default_pos  = csvd(meta["default_joint_pos"]);
    action_scale = csvd(meta["action_scale"]);
    N = static_cast<int>(joint_names.size());
    if (default_pos.size() != joint_names.size() ||
        action_scale.size() != joint_names.size()) {
        ROS_FATAL("[ee_hold] Metadata size mismatch: joints=%zu default_pos=%zu action_scale=%zu",
                  joint_names.size(), default_pos.size(), action_scale.size());
        return 1;
    }

    obs_joint_pos.assign(N, 0.0);
    obs_joint_abs_pos.assign(N, 0.0);
    obs_joint_vel.assign(N, 0.0);
    obs_actions.assign(N, 0.0);

    ROS_INFO("=== ee_hold_default_node ===");
    ROS_INFO("  joints   : %d", N);
    ROS_INFO("  obs_dim  : read from model at startup");
    ROS_INFO("  Joint  |  default_pos  |  action_scale");
    for (int i = 0; i < N; ++i)
        ROS_INFO("  [%2d] %-28s  %+.4f  %.4f",
            i, joint_names[i].c_str(),
            i < (int)default_pos.size()  ? default_pos[i]  : 0.0,
            i < (int)action_scale.size() ? action_scale[i] : 0.0);

    ROS_INFO("[ee_hold] Loading ONNX model...");
    try {
        sess.reset(new Ort::Session(env, model_path.c_str(), Ort::SessionOptions{}));
    } catch (const std::exception& e) {
        ROS_FATAL("[ee_hold] ONNX load failed: %s", e.what());
        return 1;
    }
    {
        Ort::TypeInfo type_info = sess->GetInputTypeInfo(0);
        auto shape_info = type_info.GetTensorTypeAndShapeInfo();
        auto shape = shape_info.GetShape();
        obs_dim = static_cast<int>(shape.back());
    }
    ROS_INFO("[ee_hold] ONNX model loaded. obs_dim=%d. Starting immediately (no pose required).", obs_dim);
    if (obs_dim < 3 * N) {
        ROS_FATAL("[ee_hold] ONNX obs_dim=%d is smaller than required 3*N=%d", obs_dim, 3 * N);
        return 1;
    }

    {
        Ort::AllocatorWithDefaultOptions alloc;
        in_names.push_back(sess->GetInputNameAllocated(0,  alloc).release());
        out_names.push_back(sess->GetOutputNameAllocated(0, alloc).release());
    }

    pub_cmd          = nh.advertise<sensor_msgs::JointState>("/subscriber_controller/desired_state", 1);
    pub_in_joint_pos = nh.advertise<sensor_msgs::JointState>("/policy_input/joint_pos",    1);
    pub_in_joint_vel = nh.advertise<sensor_msgs::JointState>("/policy_input/joint_vel",    1);
    pub_in_actions   = nh.advertise<sensor_msgs::JointState>("/policy_input/actions",      1);
    pub_in_obs       = nh.advertise<std_msgs::Float64MultiArray>("/policy_input/obs_vector", 1);
    pub_out_raw      = nh.advertise<sensor_msgs::JointState>("/policy_output/actions_raw",    1);
    pub_out_scaled   = nh.advertise<sensor_msgs::JointState>("/policy_output/actions_scaled", 1);

    ros::Subscriber sub_js = nh.subscribe("/joint_states", 1, jsCB);
    ros::Timer timer = nh.createTimer(ros::Duration(0.02), timerCB);

    ROS_INFO("[ee_hold] Ready. Running at 50 Hz.");
    ros::spin();
    return 0;
}
