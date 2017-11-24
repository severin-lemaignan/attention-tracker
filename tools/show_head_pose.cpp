#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <iomanip>

#include <boost/program_options.hpp>

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

    namedWindow("headpose");

    string video_file;

    po::positional_options_description p;
    p.add("video", 1);

    po::options_description desc("Allowed options");
    desc.add_options()
            ("help,h", "produces help message")
            ("version,v", "shows version and exits")
            ("model", po::value<string>(), "dlib's trained face model")
            ("video", po::value<string>(), "video to process. If omitted, uses the first webcam")
            ;

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv)
                        .options(desc)
                        .positional(p)
                        .run(), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << argv[0] << " " << STR(GAZR_VERSION) << "\n\n" << desc << "\n";
        return 1;
    }

    if (vm.count("version")) {
        cout << argv[0] << " " << STR(GAZR_VERSION) << "\n";
        return 0;
    }

    if (vm.count("model") == 0) {
        cout << "You must specify the path to a trained dlib's face model\n"
             << "with the option --model." << endl;
        return 1;
    }

    auto estimator = HeadPoseEstimation(vm["model"].as<string>());


    // Configure the video capture
    // ===========================

    VideoCapture video_in;

    if (vm.count("video") == 0) {
        video_in = VideoCapture(0);
        video_in.set(CV_CAP_PROP_FRAME_WIDTH, 640);
        video_in.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
    }
    else {
        video_in = VideoCapture(vm["video"].as<string>());
    }

    // adjust for your webcam!
    estimator.focalLength = 500;

    if(!video_in.isOpened()) {
        cerr << "Couldn't open camera/video file" << endl;
        return 1;
    }


    while(true) {
        auto ok = video_in.read(frame);
        if (!ok) break;

        auto t_start = getTickCount();

        auto all_features = estimator.update(frame);
        auto poses = estimator.poses();

        auto t_end = getTickCount();
        cout << "Processing time for this frame: " << (t_end-t_start) / getTickFrequency() * 1000. << "ms" << endl;

        imshow("headpose", estimator.drawDetections(frame, all_features, poses));
        if (waitKey(10) >= 0) break;

    }
}



