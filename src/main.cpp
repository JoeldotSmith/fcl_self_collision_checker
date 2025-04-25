#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.h>
#include <urdf/model.h>
#include <fcl/fcl.h>
#include <fcl/geometry/bvh/BVH_model.h>
#include <fcl/geometry/bvh/BVH_internal.h>
#include "sensor_msgs/msg/joint_state.hpp"
#include <fcl/geometry/shape/sphere.h>
#include <fcl/geometry/shape/box.h>
#include <fcl/geometry/shape/triangle_p.h>
#include <fcl/geometry/shape/cylinder.h>
#include <fcl/geometry/geometric_shape_to_BVH_model.h>
#include <visualization_msgs/msg/marker.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <resource_retriever/retriever.h>
#include <unordered_map>
#include <string>
#include <memory>
#include <fstream>
#include <stdexcept>

using fcl::CollisionObjectd;
using fcl::CollisionRequestd;
using fcl::CollisionResultd;
using fcl::Sphered;
using fcl::Cylinderd;
using fcl::BVHModel;
using fcl::OBBRSSd;

struct LinkCollision {
    std::shared_ptr<CollisionObjectd> object;
    Eigen::Isometry3d local_transform; 
};

class FCLRobotModel {
public:
    FCLRobotModel(const std::string& urdf_path, rclcpp::Logger logger) : logger_(logger) {
        urdf::Model model;
        if (!model.initFile(urdf_path)) {
            throw std::runtime_error("Failed to load URDF file");
        }

        for (const auto& link_pair : model.links_) {
            const auto& link = link_pair.second;
            if (!link->collision || !link->collision->geometry)
                continue;

            std::shared_ptr<CollisionObjectd> obj;
            Eigen::Isometry3d tf = Eigen::Isometry3d::Identity();
            tf.translation() << link->collision->origin.position.x,
                                link->collision->origin.position.y,
                                link->collision->origin.position.z;
            tf.linear() = Eigen::Quaterniond(
                link->collision->origin.rotation.w,
                link->collision->origin.rotation.x,
                link->collision->origin.rotation.y,
                link->collision->origin.rotation.z
            ).toRotationMatrix();

            if (auto sphere = std::dynamic_pointer_cast<urdf::Sphere>(link->collision->geometry)) {
                auto fcl_sphere = std::make_shared<Sphered>(sphere->radius);
                obj = std::make_shared<CollisionObjectd>(fcl_sphere);
            }
            else if (auto cylinder = std::dynamic_pointer_cast<urdf::Cylinder>(link->collision->geometry)) {
                auto fcl_cyl = std::make_shared<Cylinderd>(cylinder->radius, cylinder->length);
                obj = std::make_shared<CollisionObjectd>(fcl_cyl);
            }
            else if (auto box = std::dynamic_pointer_cast<urdf::Box>(link->collision->geometry)) {
                auto fcl_box = std::make_shared<fcl::Boxd>(box->dim.x, box->dim.y, box->dim.z);
                obj = std::make_shared<CollisionObjectd>(fcl_box);
            }
            else if (auto mesh = std::dynamic_pointer_cast<urdf::Mesh>(link->collision->geometry)) {
                std::string resolved_path = resolvePackagePath(mesh->filename);
            
                static MeshCache mesh_cache;
                auto cached = mesh_cache.find(resolved_path);
                if (cached != mesh_cache.end()) {
                    obj = std::make_shared<CollisionObjectd>(cached->second);
                    RCLCPP_INFO(rclcpp::get_logger("FCLRobotModel"), "Loaded mesh for link: %s", link->name.c_str());
                } else {
                    auto fcl_mesh = loadMesh(resolved_path);
                    if (fcl_mesh) {
                        mesh_cache[resolved_path] = fcl_mesh;
                        obj = std::make_shared<CollisionObjectd>(fcl_mesh);
                        RCLCPP_INFO(rclcpp::get_logger("FCLRobotModel"), "Loaded mesh for link: %s", link->name.c_str());
                    } else {
                        RCLCPP_ERROR(rclcpp::get_logger("FCLRobotModel"),
                                "Failed to load mesh: %s", resolved_path.c_str());
                        continue;
                    }
                }
            }
            else {
                RCLCPP_WARN(rclcpp::get_logger("FCLRobotModel"),
                        "Unsupported collision geometry for link: %s",
                        link->name.c_str());
                continue;
            }

            LinkCollision lc = {obj, tf};
            links_[link->name] = lc;
        }
        ignore_pairs_.insert({"logo_link", "waist_support_link"});
        ignore_pairs_.insert({"pelvis_contour_link", "waist_support_link"});
        ignore_pairs_.insert({"logo_link", "pelvis_contour_link"});
        ignore_pairs_.insert({"left_rubber_hand", "left_wrist_yaw_link"});
        ignore_pairs_.insert({"left_rubber_hand", "left_wrist_pitch_link"});
        ignore_pairs_.insert({"right_rubber_hand", "right_wrist_yaw_link"});
        ignore_pairs_.insert({"right_rubber_hand", "right_wrist_pitch_link"});
    }

    void updateTransforms(const std::unordered_map<std::string, Eigen::Isometry3d>& link_transforms) {
        for (auto& pair : links_) {
            const std::string& link_name = pair.first;
            auto& lc = pair.second;

            if (link_transforms.count(link_name)) {
                Eigen::Isometry3d global_tf = link_transforms.at(link_name) * lc.local_transform;
                lc.object->setTransform(global_tf);
            }
        }
    }

    bool checkSelfCollision(rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub) {
        bool flag = false;
        for (auto it1 = links_.begin(); it1 != links_.end(); ++it1) {
            for (auto it2 = std::next(it1); it2 != links_.end(); ++it2) {
                if (it1->first == it2->first) continue;
                auto name_pair = std::minmax(it1->first, it2->first);
                if (ignore_pairs_.count(name_pair)) continue; 
                CollisionRequestd req;
                CollisionResultd res;
                fcl::collide(it1->second.object.get(), it2->second.object.get(), req, res);
                if (res.isCollision()) {
                    auto contact = res.getContact(0);
                    publishMarker(marker_pub, contact); 
                    RCLCPP_WARN(logger_,
                        "Collision between:\n  Link 1: %s\n  Link 2: %s",
                        it1->first.c_str(), it2->first.c_str());
                    flag = true;
                }
            }
        }
        return flag;
    }

    const std::vector<std::string> getLinkNames() const {
        std::vector<std::string> names;
        for (const auto& pair : links_) {
            names.push_back(pair.first);
        }
        return names;
    }

    void publishMarker(rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub, const fcl::Contactd& contact) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "pelvis"; 
        marker.header.stamp = rclcpp::Clock().now();
        marker.ns = "collision_points";
        marker.id = marker_id_++;
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.position.x = contact.pos[0];
        marker.pose.position.y = contact.pos[1];
        marker.pose.position.z = contact.pos[2];
        marker.scale.x = marker.scale.y = marker.scale.z = 0.02;
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        marker.color.a = 1.0;
        marker.lifetime = rclcpp::Duration::from_seconds(0.1);
        marker_pub->publish(marker);
    }

private:
    std::string resolvePackagePath(const std::string& url) {
        if (url.find("file://") == 0) {
            return url.substr(7);
        }

        if (url.find("package://") == 0) {
            const size_t prefix_len = 10;
            const size_t slash_pos = url.find('/', prefix_len);
            
            if (slash_pos == std::string::npos) {
                throw std::runtime_error("Invalid package URL format: " + url);
            }

            const std::string package_name = url.substr(prefix_len, slash_pos - prefix_len);
            const std::string relative_path = url.substr(slash_pos);

            try {
                const std::string package_path = ament_index_cpp::get_package_share_directory(package_name);
                return package_path + relative_path;
            } catch (const std::runtime_error& e) {
                throw std::runtime_error("Package not found: " + package_name);
            }
        }

        return url;
    }

    std::shared_ptr<BVHModel<OBBRSSd>> loadMesh(const std::string& file_path) {
        auto model = std::make_shared<BVHModel<OBBRSSd>>();
        model->beginModel();

        std::ifstream file(file_path, std::ios::binary);
        if (!file.is_open()) {
            RCLCPP_ERROR(rclcpp::get_logger("FCLRobotModel"), 
                       "Failed to open mesh file: %s", file_path.c_str());
            return nullptr;
        }

        char header[80];
        file.read(header, 80);
        
        uint32_t tri_count;
        file.read(reinterpret_cast<char*>(&tri_count), 4);

        std::vector<fcl::Vector3d> vertices;
        vertices.reserve(tri_count * 3);

        for (uint32_t i = 0; i < tri_count; ++i) {
            file.seekg(12 + 12*3 + 2, std::ios::cur);
            fcl::Vector3d v[3];

            for (int j = 0; j < 3; ++j) {
                file.read(reinterpret_cast<char*>(&v[j][0]), sizeof(float));
                file.read(reinterpret_cast<char*>(&v[j][1]), sizeof(float));
                file.read(reinterpret_cast<char*>(&v[j][2]), sizeof(float));
            }

            vertices.insert(vertices.end(), {v[0], v[1], v[2]});
            model->addTriangle(v[0], v[1], v[2]);
        }

        for (const auto& v : vertices) {
            model->addVertex(v);
        }

        model->endModel();
        return model;
    }

    using MeshCache = std::unordered_map<std::string, std::shared_ptr<BVHModel<OBBRSSd>>>;
    int marker_id_ = 0;
    rclcpp::Logger logger_;
    std::unordered_map<std::string, LinkCollision> links_;
    std::set<std::pair<std::string, std::string>> ignore_pairs_;
};

class CollisionChecker : public rclcpp::Node {
    public:
        CollisionChecker() : Node("collision_checker") {
            marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("collision_markers", 10);
            
            // runs at about 30Hz
            joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>( "/joint_states", 10, std::bind(&CollisionChecker::updateTransforms, this, std::placeholders::_1));
            joint_pub_ = this->create_publisher<sensor_msgs::msg::JointState>( "/joint_states_cleaned", 10);
            tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
            tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_, this, false);
            
            try {
                fcl_model_ = std::make_shared<FCLRobotModel>(
                    "unitree_ws/src/fcl_self_collision_checker/urdf/g1_29dof_lock_waist.urdf",
                    this->get_logger()
                );
            } catch (const std::exception& e) {
                RCLCPP_FATAL(get_logger(), "Failed to initialize robot model: %s", e.what());
                rclcpp::shutdown();
            }
            RCLCPP_INFO(this->get_logger(), "CollisionChecker node started");
        }
    
    private:
        // depreciated
        // void benchmark() {
        //     static size_t count = 0;
        //     static double mean = 0.0;
        //     static double m2 = 0.0;
        
        //     auto start = std::chrono::high_resolution_clock::now();
        //     updateTransforms();
        //     auto end = std::chrono::high_resolution_clock::now();
        //     auto duration = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
        
        //     count++;
        //     double delta = duration - mean;
        //     mean += delta / count;
        //     m2 += delta * (duration - mean);
        
        //     if (count > 1) {
        //         double variance = m2 / (count - 1);
        //         double stddev = std::sqrt(variance);
        //         double ci95 = 1.96 * stddev / std::sqrt(count);
        //         RCLCPP_INFO(this->get_logger(), "n: %zu | Mean: %.3f ms | StdDev: %.3f ms | 95%% CI: ±%.3f ms", count, mean, stddev, ci95);
        //     } else {
        //         RCLCPP_INFO(this->get_logger(), "Collecting data... First sample: %.3f ms", duration);
        //     }
        // }

        // runs at about 385Hz
        void updateTransforms(sensor_msgs::msg::JointState::SharedPtr msg) {
            std::unordered_map<std::string, Eigen::Isometry3d> link_transforms;
            const std::string target_frame = "pelvis";
    
            for (const auto& link_name : fcl_model_->getLinkNames()) {
                try {
                    geometry_msgs::msg::TransformStamped transform_stamped =
                        tf_buffer_->lookupTransform(target_frame, link_name, tf2::TimePointZero);
                    
                    Eigen::Isometry3d transform = tf2::transformToEigen(transform_stamped.transform);
                    link_transforms[link_name] = transform;
                } catch (const tf2::TransformException& ex) {
                    RCLCPP_WARN(get_logger(), "Could not transform %s to %s: %s",
                        target_frame.c_str(), link_name.c_str(), ex.what());
                    continue;
                }
            }
    
            try {
                fcl_model_->updateTransforms(link_transforms);
            } catch (const std::exception& e) {
                RCLCPP_ERROR(get_logger(), "Transform update failed: %s", e.what());
                return;
            }
    
            if (fcl_model_->checkSelfCollision(marker_pub_)) {
                RCLCPP_WARN(get_logger(), "Self-collision detected!");
            } else {
                RCLCPP_INFO(get_logger(), "No collision.");
                joint_pub_->publish(*msg);
            }
        }
    
        std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
        std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
        rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
        rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_pub_;
        rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
        std::shared_ptr<FCLRobotModel> fcl_model_;
    };

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CollisionChecker>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}