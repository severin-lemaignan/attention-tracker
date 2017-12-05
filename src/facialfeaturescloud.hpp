#include <vector>

#include <opencv2/core/core.hpp>
#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>

#include <image_geometry/pinhole_camera_model.h>

#include "head_pose_estimation.hpp"

/**
 * This class is heavily based on https://github.com/ros-perception/image_pipeline/blob/indigo/depth_image_proc/src/nodelets/point_cloud_xyzrgb.cpp
 */
class FacialFeaturesPointCloudPublisher {

public:
    FacialFeaturesPointCloudPublisher(ros::NodeHandle& rosNode,
                                      const std::string& prefix,
                                      const std::string& model);

    void imageCb(const sensor_msgs::ImageConstPtr& rgb_msg,
                 const sensor_msgs::ImageConstPtr& depth_msg,
                 const sensor_msgs::CameraInfoConstPtr& depth_camerainfo);
private:

    template<typename T>
    void makeFeatureCloud(const std::vector<cv::Point> points2d,
                          const sensor_msgs::ImageConstPtr& depth_msg,
                          sensor_msgs::PointCloud2Ptr& cloud_msg);

    image_geometry::PinholeCameraModel cameramodel;

    cv::Mat inputImage;

    tf::TransformBroadcaster br;
    tf::Transform transform;

    HeadPoseEstimation estimator;

    // prefix prepended to TF frames generated for each frame
    std::string facePrefix;

    // Subscriptions
    /////////////////////////////////////////////////////////
    std::shared_ptr<image_transport::ImageTransport> rgb_it_;
    std::shared_ptr<image_transport::ImageTransport> depth_it_;
    image_transport::SubscriberFilter sub_depth_;
    image_transport::SubscriberFilter sub_rgb_;
    message_filters::Subscriber<sensor_msgs::CameraInfo> sub_info_;

    // Publishers
    /////////////////////////////////////////////////////////
    ros::Publisher nb_detected_faces_pub;

    ros::Publisher facial_features_pub;
    
#ifdef HEAD_POSE_ESTIMATION_DEBUG
    image_transport::Publisher pub;
#endif

    // typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo> SyncPolicy;
    typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo> ExactSyncPolicy;
    // typedef message_filters::Synchronizer<SyncPolicy> Synchronizer;
    typedef message_filters::Synchronizer<ExactSyncPolicy> ExactSynchronizer;
    // std::shared_ptr<Synchronizer> sync_;
    std::shared_ptr<ExactSynchronizer> exact_sync_;
};

