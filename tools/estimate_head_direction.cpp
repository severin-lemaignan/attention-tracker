#include <opencv2/opencv.hpp>

#include <iostream>
#include <iomanip>

#include <boost/program_options.hpp>

#include "LinearMath/Matrix3x3.h"

#include "../src/head_pose_estimation.hpp"

#define STR_EXPAND(tok) #tok
#define STR(tok) STR_EXPAND(tok)

using namespace std;
using namespace cv;
namespace po = boost::program_options;

inline double todeg(double rad) {
    return rad * 180 / M_PI;
}


int main(int argc, char **argv)
{
    Mat frame;
    bool show_frame = false;
    bool use_camera = false;

    po::positional_options_description p;
    p.add("image", 1);

    po::options_description desc("Allowed options");
    desc.add_options()
            ("help,h", "produce help message")
            ("version,v", "shows version and exits")
            ("show,s", "display the image with gaze estimation")
            ("model", po::value<string>(), "dlib's trained face model")
            ("image", po::value<string>(), "image to process (png, jpg)")
            ;

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv)
                        .options(desc)
                        .positional(p)
                        .run(), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << argv[0] << " " <<  STR(GAZR_VERSION) << "\n\n" << desc << "\n";
        return 1;
    }

    if (vm.count("version")) {
        cout << argv[0] << " " << STR(GAZR_VERSION) << "\n";
        return 0;
    }

    if (vm.count("show")) {
        show_frame = true;
    }

    if (vm.count("model") == 0) {
        cout << "You must specify the path to a trained dlib's face model\n"
             << "with the option --model." << endl;
        return 1;
    }

    if (vm.count("image") == 0) {
        use_camera = true;
    }

    auto estimator = HeadPoseEstimation(vm["model"].as<string>());

    VideoCapture video_in;

    if (use_camera) {
        video_in = VideoCapture(0);

        // adjust for your webcam!
        video_in.set(CV_CAP_PROP_FRAME_WIDTH, 640);
        video_in.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
        estimator.focalLength = 500;
        estimator.opticalCenterX = 320;
        estimator.opticalCenterY = 240;

        if(!video_in.isOpened()) {
            cerr << "Couldn't open camera" << endl;
            return 1;
        }
    }

    else {
        auto image = vm["image"].as<string>();

#ifdef OPENCV3
        frame = imread(image,IMREAD_COLOR);
#else
        frame = imread(image,CV_LOAD_IMAGE_COLOR);
#endif

        resize(frame, frame, Size(0,0), 0.2,0.2);

        estimator.focalLength = 85.0 / 22.3 * frame.size().width;
    }

    while(true) {
        if(use_camera) {
            auto ok = video_in.read(frame);
            if (!ok) break;
        }


        auto all_features = estimator.update(frame);


        auto poses = estimator.poses();

        int i = 0;
        cout << "{";

        for(auto pose : poses) {



            pose = pose.inv();

            double raw_yaw, raw_pitch, raw_roll;
            tf::Matrix3x3 mrot(
                    pose(0,0), pose(0,1), pose(0,2),
                    pose(1,0), pose(1,1), pose(1,2),
                    pose(2,0), pose(2,1), pose(2,2));
            mrot.getRPY(raw_roll, raw_pitch, raw_yaw);

            raw_roll = raw_roll - M_PI/2;
            raw_yaw = raw_yaw + M_PI/2;

            double yaw, pitch, roll;

            roll = raw_pitch;
            yaw = raw_yaw;
            pitch = -raw_roll;

            cout << "\"face_" << i << "\":";
            cout << setprecision(1) << fixed << "{\"yaw\":" << todeg(yaw) << ", \"pitch\":" << todeg(pitch) << ", \"roll\":" << todeg(roll) << ",";
            cout << setprecision(4) << fixed << "\"x\":" << pose(0,3) << ", \"y\":" << pose(1,3) << ", \"z\":" << pose(2,3) << "},";

            i++;
        }
        cout << "}\n" << flush;

        if (show_frame) {
            imshow("headpose", estimator.drawDetections(frame, all_features, poses));
            if(use_camera) {
                waitKey(10);
            }
            else {
                while(waitKey(10) != 1048603) {}
                break;
            }
        }
    }

}


