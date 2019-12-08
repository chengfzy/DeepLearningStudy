#include <chrono>
#include <iostream>
#include <memory>
#include <vector>
//#include "Darknet.h"
#include "gflags/gflags.h"
#include "opencv2/core.hpp"
#include "opencv2/freetype.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;
//using namespace torch::jit;

DEFINE_string(weightFile, "../../../temp/yolov3.weights", "yolov3 weight file");
DEFINE_string(configFile, "../../../temp/yolov3.cfg", "yolov3 config file");
DEFINE_string(imageFile, "", "image file");
DEFINE_int32(imageSize, 608, "input image size for yolov3");
DEFINE_int32(classesNum, 80, "class number to detect");

/*
// draw detection to image
void drawDetection(cv::Mat& image, int classId, const float& conf, int left, int top, int right, int bottom) {
    // detection box
    rectangle(image, Point(left, top), Point(right, bottom), Scalar(0, 255, 0), 1);

    // label
    string label = to_string(classId) + format(": %.3f", conf);
    Ptr<freetype::FreeType2> font = freetype::createFreeType2();
    font->loadFontData("../../config/NotoSansCJK-Regular.ttc", 0);
    int baseLine{0};
    Size labelSize = font->getTextSize(label, 15, FILLED, &baseLine);
    top = max(top, labelSize.height);
    rectangle(image, Point(left, top - labelSize.height), Point(left + labelSize.width, top + baseLine), Scalar(255),
              FILLED);
    font->putText(image, label, Point(left, top), 15, Scalar(0), FILLED, LINE_AA, true);
}
*/

int main(int argc, char* argv[]) {
    google::ParseCommandLineFlags(&argc, &argv, true);

    /*// check CUDA is available
    torch::Device device(torch::kCPU);
    // if (torch::cuda::is_available()) {
    //    device = torch::kCUDA;
    //}

    // load model
    Darknet net(FLAGS_configFile.c_str(), &device);
    map<string, string>* info = net.get_net_info();
    info->operator[]("height") = to_string(FLAGS_imageSize);
    net.load_weights(FLAGS_weightFile.c_str());
    // net.to(device);  // to device
    //torch::NoGradGuard noGrad;
    //net.eval();  // eval mode*/

    // load image
    Mat rawImg = cv::imread(FLAGS_imageFile, cv::IMREAD_UNCHANGED);
    // convert to BGR color and resize, and to float in (0,1)
    //Mat inputImg;
    //cvtColor(rawImg, inputImg, COLOR_BGR2RGB);
    //resize(inputImg, inputImg, Size(FLAGS_imageSize, FLAGS_imageSize));
    //inputImg.convertTo(inputImg, CV_32F, 1.0 / 255);

    /*// convert image to tensor
    auto
        imgTensor = torch::CPU(torch::kFloat32).tensorFromBlob(inputImg.data, {1, FLAGS_imageSize, FLAGS_imageSize, 3});
    imgTensor = imgTensor.permute({0, 3, 1, 2});
    auto imgVar = torch::autograd::make_variable(imgTensor, false).to(device);

    // predict
    auto t0 = chrono::steady_clock::now();
    auto output = net.forward(imgVar);
    auto result = net.write_results(output, FLAGS_classesNum, 0.6, 0.4);
    auto t1 = chrono::steady_clock::now();
    cout << "used time: " << chrono::duration_cast<chrono::milliseconds>(t1 - t0).count() << endl;*/

    google::ShutDownCommandLineFlags();
    return 0;
}