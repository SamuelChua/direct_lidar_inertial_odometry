#ifndef DEGENERACY_DETECTOR_H
#define DEGENERACY_DETECTOR_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <vector>
#include <deque>
#include <unsupported/Eigen/NumericalDiff>
#include <Eigen/Dense>

class DegeneracyDetector {
public:
    DegeneracyDetector(size_t buffer_size);

    double pointToPointResidue(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_source, 
                           const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_target);

    double pointToLineDistance(const Eigen::Vector3d& point, const Eigen::Vector3d& line_point1, const Eigen::Vector3d& line_point2);
                           
    bool fitLineRANSAC(const std::vector<Eigen::Vector3d>& points, Eigen::Vector3d& best_line_point1, Eigen::Vector3d& best_line_point2, double threshold, int iterations);                    

    bool planeFromPoints(const std::vector<Eigen::Vector3d>& points, Eigen::Vector3d& planeNormal, double& planeD, double planeFitThreshold);
    
    // Function to calculate normalized eigenvalues for degeneracy detection
    void calculateNormalizedEigenvalues(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& scan_points, 
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& map_cloud, 
    const Eigen::Matrix<double, 6, 1>& current_pose);

    // Function to detect degeneracy in the point cloud
    bool detectDegeneracy(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                      const pcl::PointCloud<pcl::PointXYZ>::Ptr& map_cloud,
                      const Eigen::Matrix<double, 6, 1>& current_pose);


private:
    size_t buffer_size;
    std::deque<std::string> status_buffer;
    std::deque<std::string> decision_buffer;
    std::vector<float> normalized_eigenvalues;
    Eigen::MatrixXd eigenvectors;

    // Function to update the status buffer
    void updateStatusBuffer(bool is_degenerate);
    
    void updateDecisionBuffer();

    // Function to get the current status based on the status buffer
    std::string getCurrentStatus();

    // Function to perform a chi-squared test on eigenvalues
    bool chiSquaredTest(const std::vector<float>& eigenvalues);
};

#endif // DEGENERACY_DETECTOR_H
