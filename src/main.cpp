#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <urdf/model.h>
#include <fcl/fcl.h>
#include <fcl/geometry/bvh/BVH_model.h>
#include <fcl/geometry/bvh/BVH_internal.h>
#include <fcl/geometry/shape/sphere.h>
#include <fcl/geometry/shape/triangle_p.h>
#include <fcl/geometry/shape/cylinder.h>
#include <fcl/geometry/geometric_shape_to_BVH_model.h>
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

// Mesh cache type
using MeshCache = std::unordered_map<std::string, std::shared_ptr<BVHModel<OBBRSSd>>>;

struct LinkCollision {
    std::shared_ptr<CollisionObjectd> object;
    std::string parent_joint;
    Eigen::Isometry3d local_transform;
};

class FCLRobotModel {
public:
    FCLRobotModel(const std::string& urdf_path) {
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
            else if (auto mesh = std::dynamic_pointer_cast<urdf::Mesh>(link->collision->geometry)) {
                std::string resolved_path = resolvePackagePath(mesh->filename);
                
                // Check cache first
                static MeshCache mesh_cache;
                auto cached = mesh_cache.find(resolved_path);
                if (cached != mesh_cache.end()) {
                    obj = std::make_shared<CollisionObjectd>(cached->second);
                } else {
                    auto fcl_mesh = loadMesh(resolved_path);
                    if (fcl_mesh) {
                        mesh_cache[resolved_path] = fcl_mesh;
                        obj = std::make_shared<CollisionObjectd>(fcl_mesh);
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

            LinkCollision lc = {obj, link->parent_joint ? link->parent_joint->name : "", tf};
            links_[link->name] = lc;
        }
    }

    void updateTransforms(const std::unordered_map<std::string, double>& joint_positions) {
        for (auto& pair : links_) {
            auto& lc = pair.second;
            double angle = 0.0;
            if (joint_positions.count(lc.parent_joint))
                angle = joint_positions.at(lc.parent_joint);

            Eigen::Isometry3d joint_tf = Eigen::Isometry3d::Identity();
            joint_tf.linear() = Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitZ()).toRotationMatrix();

            lc.object->setTransform(joint_tf * lc.local_transform);
        }
    }

    bool checkSelfCollision() {
        for (auto it1 = links_.begin(); it1 != links_.end(); ++it1) {
            for (auto it2 = std::next(it1); it2 != links_.end(); ++it2) {
                CollisionRequestd req;
                CollisionResultd res;
                fcl::collide(it1->second.object.get(), it2->second.object.get(), req, res);
                if (res.isCollision())
                    return true;
            }
        }
        return false;
    }

private:
    std::string resolvePackagePath(const std::string& url) {
        // Handle file:// URLs
        if (url.find("file://") == 0) {
            return url.substr(7);
        }

        // Handle package:// URLs
        if (url.find("package://") == 0) {
            const size_t prefix_len = 10; // "package://" length
            const size_t slash_pos = url.find('/', prefix_len);
            
            if (slash_pos == std::string::npos) {
                throw std::runtime_error("Invalid package URL format - missing '/': " + url);
            }

            const std::string package_name = url.substr(prefix_len, slash_pos - prefix_len);
            const std::string relative_path = url.substr(slash_pos);

            try {
                const std::string package_path = ament_index_cpp::get_package_share_directory(package_name);
                return package_path + relative_path;
            } catch (const std::runtime_error& e) {
                throw std::runtime_error("Package not found: " + package_name + " (while resolving: " + url + ")");
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

        // Read STL header
        char header[80];
        file.read(header, 80);
        
        // Read triangle count
        uint32_t tri_count;
        file.read(reinterpret_cast<char*>(&tri_count), 4);

        std::vector<fcl::Vector3d> vertices;
        vertices.reserve(tri_count * 3);

        // Read triangles
        for (uint32_t i = 0; i < tri_count; ++i) {
            file.seekg(12 + 12*3 + 2, std::ios::cur);
            fcl::Vector3d v[3];

            for (int j = 0; j < 3; ++j) {
                file.read(reinterpret_cast<char*>(&v[j][0]), sizeof(float));
                file.read(reinterpret_cast<char*>(&v[j][1]), sizeof(float));
                file.read(reinterpret_cast<char*>(&v[j][2]), sizeof(float));
            }

            // Add vertices and triangle
            vertices.insert(vertices.end(), {v[0], v[1], v[2]});
            model->addTriangle(v[0], v[1], v[2]);
        }

        for (const auto& v : vertices) {
            model->addVertex(v);
        }

        model->endModel();
        return model;
    }

    std::unordered_map<std::string, LinkCollision> links_;
};

class CollisionChecker : public rclcpp::Node {
public:
    CollisionChecker() : Node("collision_checker") {
        joint_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", 10,
            std::bind(&CollisionChecker::jointCallback, this, std::placeholders::_1));
        
        joint_pub_ = this->create_publisher<sensor_msgs::msg::JointState>(
            "/joint_states_cleaned", 10);
        
        try {
            fcl_model_ = std::make_shared<FCLRobotModel>(
                "unitree_ws/src/fcl_self_collision_checker/urdf/g1_29dof_lock_waist.urdf");
        } catch (const std::exception& e) {
            RCLCPP_FATAL(get_logger(), "Failed to initialize robot model: %s", e.what());
            rclcpp::shutdown();
        }
    }

private:
    void jointCallback(const sensor_msgs::msg::JointState::SharedPtr msg) {
        std::unordered_map<std::string, double> joints;
        for (size_t i = 0; i < msg->name.size(); ++i)
            joints[msg->name[i]] = msg->position[i];

        try {
            fcl_model_->updateTransforms(joints);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "Transform update failed: %s", e.what());
            return;
        }

        if (fcl_model_->checkSelfCollision()) {
            // RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "Self-collision detected!");
            RCLCPP_WARN(this->get_logger(), "Self-collision detected!");
        } else {
            RCLCPP_INFO(this->get_logger(), "No collision.");
            joint_pub_->publish(*msg);
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_pub_;
    std::shared_ptr<FCLRobotModel> fcl_model_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CollisionChecker>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}