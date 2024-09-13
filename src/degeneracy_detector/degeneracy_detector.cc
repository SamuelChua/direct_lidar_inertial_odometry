#include "degeneracy_detector/degeneracy_detector.h"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/centroid.h>
#include <cmath>
#include <ros/ros.h>  
#include <random>
 

DegeneracyDetector::DegeneracyDetector(size_t buffer_size) : buffer_size(buffer_size) {
    // Initialize the status buffer with "Normal" states
    for (size_t i = 0; i < buffer_size; ++i) {
        status_buffer.push_back("Normal");
    }
}

double DegeneracyDetector::pointToPointResidue(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_source, 
                           const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_target) {
    if (cloud_source->size() != cloud_target->size()) {
        throw std::runtime_error("Source and target point clouds must have the same size");
    }

    double residue = 0.0;

    // Loop through all points in the source and target point clouds
    for (size_t i = 0; i < cloud_source->size(); ++i) {
        const auto& p_src = cloud_source->points[i];
        const auto& p_tgt = cloud_target->points[i];

        // Compute the squared distance between corresponding points
        Eigen::Vector3d source_point(p_src.x, p_src.y, p_src.z);
        Eigen::Vector3d target_point(p_tgt.x, p_tgt.y, p_tgt.z);

        double dist_sq = (source_point - target_point).squaredNorm();
        residue += dist_sq;
    }

    // Return the average residue
    return residue / static_cast<double>(cloud_source->size());
}

double DegeneracyDetector::pointToLineDistance(const Eigen::Vector3d& point, const Eigen::Vector3d& line_point1, const Eigen::Vector3d& line_point2) {
    Eigen::Vector3d line_direction = line_point2 - line_point1;
    Eigen::Vector3d cross_product = (point - line_point1).cross(line_direction);
    return cross_product.norm() / line_direction.norm();
}

// RANSAC function to fit a line to a set of points
bool DegeneracyDetector::fitLineRANSAC(const std::vector<Eigen::Vector3d>& points, Eigen::Vector3d& best_line_point1, Eigen::Vector3d& best_line_point2, double threshold, int iterations) {
    if (points.size() < 2) {
        std::cerr << "At least 2 points are required to fit a line." << std::endl;
        return false;
    }

    size_t best_inlier_count = 0;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, points.size() - 1);

    for (int i = 0; i < iterations; ++i) {
        // Randomly sample two points
        int idx1 = dist(gen);
        int idx2 = dist(gen);
        while (idx1 == idx2) {
            idx2 = dist(gen);
        }

        Eigen::Vector3d line_point1 = points[idx1];
        Eigen::Vector3d line_point2 = points[idx2];

        // Count inliers
        size_t inlier_count = 0;
        for (const auto& point : points) {
            double distance = pointToLineDistance(point, line_point1, line_point2);
            if (distance < threshold) {
                ++inlier_count;
            }
        }

        // Update the best line if we have more inliers
        if (inlier_count > best_inlier_count) {
            best_inlier_count = inlier_count;
            best_line_point1 = line_point1;
            best_line_point2 = line_point2;
        }
    }

    return best_inlier_count > 0;
}

bool DegeneracyDetector::planeFromPoints(const std::vector<Eigen::Vector3d>& points, Eigen::Vector3d& planeNormal, double& planeD, double planeFitThreshold) {
    if (points.size() < 3) {
        std::cerr << "At least 3 points are required to fit a plane." << std::endl;
        return false;
    }
 
    // Compute the centroid of the points
    Eigen::Vector3d centroid(0, 0, 0);
    for (const auto& point : points) {
        centroid += point;
    }
    centroid /= points.size();

    // Compute the covariance matrix
    double xx = 0.0, xy = 0.0, xz = 0.0;
    double yy = 0.0, yz = 0.0, zz = 0.0;

    for (const auto& point : points) {
        Eigen::Vector3d r = point - centroid;
        xx += r.x() * r.x();
        xy += r.x() * r.y();
        xz += r.x() * r.z();
        yy += r.y() * r.y();
        yz += r.y() * r.z();
        zz += r.z() * r.z();
    }

    // Find the determinant for each case
    double detX = yy * zz - yz * yz;
    double detY = xx * zz - xz * xz;
    double detZ = xx * yy - xy * xy;

  // Find the maximum determinant to select the best axis for the normal direction
    double detMax = std::max({detX, detY, detZ});

    if (detMax <= planeFitThreshold) {
        //std::cerr << "The points do not span a valid plane." << std::endl;
        return false;
    }

    Eigen::Vector3d normal(0, 0, 0);

    if (detMax == detX) {
        normal.x() = detX;
        normal.y() = xz * yz - xy * zz;
        normal.z() = xy * yz - xz * yy;
    } else if (detMax == detY) {
        normal.x() = xz * yz - xy * zz;
        normal.y() = detY;
        normal.z() = xy * xz - yz * xx;
    } else if (detMax == detZ) {
        normal.x() = xy * yz - xz * yy;
        normal.y() = xy * xz - yz * xx;
        normal.z() = detZ;
    }

    // Normalize the normal vector
    planeNormal = normal.normalized();

    // Calculate D for the plane equation
    planeD = -planeNormal.dot(centroid);

    return true;
}

template <typename Scalar>
struct DegeneracyFunctor {
    using InputType = Eigen::Matrix<Scalar, 3, 1>;  // 3-DOF input (translation: x, y, z)
    using ValueType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;  // Residuals (d_l)
    using JacobianType = Eigen::Matrix<Scalar, Eigen::Dynamic, 3>;  // Jacobian with respect to (x, y, z)

    pcl::PointCloud<pcl::PointXYZ>::Ptr scan_cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr map_cloud;
    pcl::KdTreeFLANN<pcl::PointXYZ>& kdtree;

    DegeneracyFunctor(pcl::PointCloud<pcl::PointXYZ>::Ptr scan, pcl::PointCloud<pcl::PointXYZ>::Ptr m, pcl::KdTreeFLANN<pcl::PointXYZ>& kd)
        : scan_cloud(scan), map_cloud(m), kdtree(kd) {}

    int inputs() const { return InputType::RowsAtCompileTime; }
    int values() const { return scan_cloud->size(); }

    // Function operator to compute the residuals and optionally the Jacobian
    template <typename T1, typename T2, typename T3>
    void operator()(const T1& x, T2* f_l, T3* J_l = nullptr) const {
        f_l->resize(scan_cloud->points.size());
        if (J_l) {
            J_l->resize(scan_cloud->points.size(), 3);  // Jacobian is 3-DOF: translation only
        }

        for (size_t i = 0; i < scan_cloud->points.size(); ++i) {
            pcl::PointXYZ p_j = scan_cloud->points[i];
            int k = 10;
            std::vector<int> point_idx_search(k);
            std::vector<float> point_squared_distance(k);

            // Find the nearest points in the map cloud for plane fitting
            if (kdtree.nearestKSearch(p_j, k, point_idx_search, point_squared_distance) > 0) {
                std::vector<Eigen::Vector3d> nearest_points;
                for (size_t j = 0; j < k; ++j) {
                    nearest_points.push_back(Eigen::Vector3d(map_cloud->points[point_idx_search[j]].x,
                                                             map_cloud->points[point_idx_search[j]].y,
                                                             map_cloud->points[point_idx_search[j]].z));
                }

                // Fit a plane to the nearest points
                double planeFitThreshold = 0.05; // can vary between 0.1 and 0.01
                Eigen::Vector3d planeNormal;
                double planeD;
                if (planeFromPoints(nearest_points, planeNormal, planeD, planeFitThreshold)) {
                    Eigen::Vector3d p_j_vec(p_j.x, p_j.y, p_j.z);
                    Scalar d_p = fabs((p_j_vec.dot(planeNormal) + planeD) / planeNormal.norm());

                    (*f_l)(i) = d_p;  // Residual: point-to-plane distance

                    // Compute the Jacobian with respect to (x_t, y_t, z_t) (translation only)
                    if (J_l) {
                        (*J_l)(i, 0) = -planeNormal.x() / planeNormal.norm();  // ∂residual/∂x
                        (*J_l)(i, 1) = -planeNormal.y() / planeNormal.norm();  // ∂residual/∂y
                        (*J_l)(i, 2) = -planeNormal.z() / planeNormal.norm();  // ∂residual/∂z
                    }
                } else {
                    //ROS_WARN("Failed to fit a valid plane for point %zu", i);
                    continue;
                }
            }
        }
    }
};

void DegeneracyDetector::calculateNormalizedEigenvalues(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& scan_points, 
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& map_cloud, 
    const Eigen::Matrix<double, 6, 1>& current_pose) {

    if (map_cloud->empty()) {
        ROS_ERROR("Map cloud is empty. Cannot proceed with KDTree creation.");
        return;
    }

    // Initialize KD-Tree for nearest neighbor search
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(map_cloud);
    
    // Initialize the functor with the provided clouds
    DegeneracyFunctor<double> functor(scan_points, map_cloud, kdtree);

    // Extract only the translation part (x, y, z) from the current pose
    Eigen::Matrix<double, 3, 1> x = current_pose.head<3>();  // Extract first 3 elements (x, y, z)

    // Distance and Jacobian outputs for this scan
    Eigen::Matrix<double, Eigen::Dynamic, 1> f_l;
    Eigen::Matrix<double, Eigen::Dynamic, 3> J_l;

    // Apply the functor for the current scan to compute f_l and J_l (if needed)
    functor(x, &f_l, &J_l);

    // Compute Hessian and Eigenvalues after processing the scan
    float lambda = 0.1f;
    Eigen::VectorXd diag_matrix = (J_l.transpose() * J_l).diagonal();
    Eigen::MatrixXd matrix_to_invert = J_l.transpose() * J_l + lambda * diag_matrix.asDiagonal().toDenseMatrix();
    Eigen::MatrixXd H_l = matrix_to_invert.inverse();

    // Compute both eigenvalues and eigenvectors
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver(H_l);
    Eigen::VectorXd eigenvalues = eigen_solver.eigenvalues();
    eigenvectors = eigen_solver.eigenvectors();

    // Compute normalized eigenvalues
    float sum_of_squares = eigenvalues.head(3).squaredNorm();
    float sqrt_of_sum_of_squares = sqrt(sum_of_squares);
    normalized_eigenvalues = {
        static_cast<float>(eigenvalues(0) / sqrt_of_sum_of_squares),
        static_cast<float>(eigenvalues(1) / sqrt_of_sum_of_squares),
        static_cast<float>(eigenvalues(2) / sqrt_of_sum_of_squares)
    };
    ROS_INFO("Normalized Eigenvalues: [%f, %f, %f]", normalized_eigenvalues[0], normalized_eigenvalues[1], normalized_eigenvalues[2]);
    ROS_INFO_STREAM("Eigenvectors (first 3 columns): \n" << eigenvectors.leftCols(3));
}

bool DegeneracyDetector::detectDegeneracy(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                                          const pcl::PointCloud<pcl::PointXYZ>::Ptr& map_cloud,
                                          const Eigen::Matrix<double, 6, 1>& current_pose) {
    calculateNormalizedEigenvalues(cloud, map_cloud, current_pose);
    bool is_degenerate = chiSquaredTest(normalized_eigenvalues);

    // Update the status buffer
    updateStatusBuffer(is_degenerate);
    updateDecisionBuffer();

    // Determine the current system status
    std::string current_status = getCurrentStatus();
    ROS_INFO("Current Status: %s", current_status.c_str());

    // Return whether the system is degenerate 
    return current_status != "Normal";

}

void DegeneracyDetector::updateStatusBuffer(bool is_degenerate) {
    // Update the buffer with the new status
    std::string new_status = is_degenerate ? "Degenerate" : "Normal";
    status_buffer.push_back(new_status);

    // Remove the oldest status if the buffer exceeds the predefined size
    if (status_buffer.size() > buffer_size) {
        status_buffer.pop_front();
    }
}


void DegeneracyDetector::updateDecisionBuffer(){
    // Track fluctuations in X, Y, Z separately
    static Eigen::Vector3d prev_eigenvector_x = eigenvectors.col(0).head(3);  // Largest eigenvector's first three components
    static Eigen::Vector3d prev_eigenvector_y = eigenvectors.col(1).head(3);
    static Eigen::Vector3d prev_eigenvector_z = eigenvectors.col(2).head(3);

    Eigen::Vector3d current_eigenvector_x = eigenvectors.col(0).head(3);
    Eigen::Vector3d current_eigenvector_y = eigenvectors.col(1).head(3);
    Eigen::Vector3d current_eigenvector_z = eigenvectors.col(2).head(3);

    // Calculate the dot products to measure changes in each direction
    double similarity_x = prev_eigenvector_x.dot(current_eigenvector_x);
    double similarity_y = prev_eigenvector_y.dot(current_eigenvector_y);
    double similarity_z = prev_eigenvector_z.dot(current_eigenvector_z);

    // Threshold for determining consistency in each direction
    double threshold = 0.8;  

    std::string current_decision;

    // Detect the type of environment and store the decision in the buffer
    if (std::abs(similarity_x) < threshold || std::abs(similarity_y) < threshold) {
        current_decision = "Staircase";
        ROS_INFO("Significant fluctuations in x or y detected. Likely a staircase.");
    } else if (std::abs(similarity_z) > threshold) {
        current_decision = "Corridor";
        ROS_INFO("Z direction remains consistent. Likely a corridor.");
    } else {
        current_decision = "Unconstrained";
        ROS_INFO("Moving through a less constrained environment.");
    }

    // Update the buffer
    decision_buffer.push_back(current_decision);
    if (decision_buffer.size() > buffer_size) {
        decision_buffer.pop_front(); 
    }

    // Check buffer to determine confidence in decision
    int staircase_count = std::count(decision_buffer.begin(), decision_buffer.end(), "Staircase");
    int corridor_count = std::count(decision_buffer.begin(), decision_buffer.end(), "Corridor");

    if (staircase_count > corridor_count && staircase_count > buffer_size / 2) {
        ROS_INFO("Confidence high: Motion distortion due to aggressive movement");
    } else if (corridor_count > staircase_count && corridor_count > buffer_size / 2) {
        ROS_INFO("Confidence high: Geometric degeneracy detected");
    }

    // Update previous eigenvectors
    prev_eigenvector_x = current_eigenvector_x;
    prev_eigenvector_y = current_eigenvector_y;
    prev_eigenvector_z = current_eigenvector_z;
}

std::string DegeneracyDetector::getCurrentStatus() {
    // Determine the current status based on the status buffer
    int degenerate_count = std::count(status_buffer.begin(), status_buffer.end(), "Degenerate");

    if (degenerate_count == buffer_size) {
        return "Fully degenerate";
    } else if (degenerate_count > 0) {
        return "Start/End to degenerate";
    } else {
        return "Normal";
    }
}

bool DegeneracyDetector::chiSquaredTest(const std::vector<float>& eigenvalues) {
    const std::vector<float> expected_values{0.289f, 0.498f, 0.749f};
    const float chi_squared_critical = 0.103f;
    std::vector<float> thresholds(expected_values.size());

    // Calculate thresholds based on the provided formula
    for (size_t i = 0; i < expected_values.size(); ++i) {
        thresholds[i] = expected_values[i] - std::sqrt(chi_squared_critical * expected_values[i]);
    }

    // Check each normalized eigenvalue against its threshold
    bool is_degenerate = false;
    for (size_t i = 0; i < eigenvalues.size(); ++i) {
        if (eigenvalues[i] < thresholds[i]) {
            is_degenerate = true;
            ROS_WARN("Degeneracy detected at index %lu: Normalized Eigenvalue [%f], Threshold [%f]",
                     i, eigenvalues[i], thresholds[i]);
        }
    }
    return is_degenerate;
}

