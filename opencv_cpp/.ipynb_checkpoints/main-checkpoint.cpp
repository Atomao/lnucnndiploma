#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <chrono>
#include "inference.h"
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>

using namespace std;
using namespace cv;
using namespace std::chrono;
namespace fs = std::filesystem;

string exec(const char* cmd) {
    std::array<char, 128> buffer;
    string result;
    unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

void append_to_csv(const string& outputPath, const string& filename, double duration, double temp, double ram, double cpu) {
    ofstream file;
    file.open(outputPath, ios::out | ios::app);
    file << filename << "," << duration << "," << temp << "," << ram << "," << cpu << endl;
    file.close();
}

int main(int argc, char **argv)
{
    string inputFolderPath = "/tmp/test/video"; // Set to your folder with frames
    string outputFolderPath = "/tmp/test/video_output/"; // Set to your folder for output frames
    string csvPath = "/home/pi/exp/exp1/src/logs/system_metrics_opecv_cpp.csv"; // Path for the CSV file

    bool runOnGPU = false; // Set to false to run inference on CPU
    Inference inf("/home/pi/exp/exp1/src/yolov5nu.onnx", cv::Size(640, 640), "classes.txt", runOnGPU);

    // Ensure output directory exists
    fs::create_directories(outputFolderPath);

    // Prepare the CSV file
    ofstream csvFile(csvPath, ios::out);
    csvFile << "Filename,Inference Time (ms),Temperature (°C),Used RAM (MB),CPU Usage (%)\n";
    csvFile.close();

    // Iterate over each file in the folder
    for (const auto& entry : fs::directory_iterator(inputFolderPath)) {
        const auto& path = entry.path();
        if (path.extension() == ".jpg" || path.extension() == ".png") { // Add more extensions as needed
            Mat frame = imread(path.string());

            if (frame.empty()) {
                cerr << "Error opening image file: " << path << endl;
                continue;
            }

            auto start = high_resolution_clock::now();

            // Perform inference
            std::vector<Detection> output = inf.runInference(frame);

            auto end = high_resolution_clock::now();
            auto duration = duration_cast<milliseconds>(end - start).count();

            // Calculate and print the inference speed
            cout << "Inference time for " << path.filename() << ": " << duration << " ms" << endl;

            // Draw detections and labels on the frame
            for (const auto& detection : output) {
                rectangle(frame, detection.box, detection.color, 2);
                string label = detection.className + " " + to_string(detection.confidence).substr(0, 4);
                int baseline;
                Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
                rectangle(frame, Point(detection.box.x, detection.box.y - labelSize.height - baseline),
                          Point(detection.box.x + labelSize.width, detection.box.y),
                          detection.color, FILLED);
                putText(frame, label, Point(detection.box.x, detection.box.y - baseline),
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
            }

            // Construct output file path
            string outputPath = outputFolderPath + "/" + path.filename().string();

            // Save the processed frame
            imwrite(outputPath, frame);

            // Get system metrics
            double temp = stod(exec("vcgencmd measure_temp | cut -d '=' -f2 | cut -d \"'\" -f1"));
            double ram = stod(exec("free | grep Mem | awk '{print $3 / 1024}'"));
            double cpu = stod(exec("top -bn1 | grep 'Cpu(s)' | sed 's/.*, *\\([0-9.]*\\)%* id.*/\\1/' | awk '{print 100 - $1}'"));

            // Append metrics to CSV
            append_to_csv(csvPath, path.filename(), duration, temp, ram, cpu);
        }
    }

    return 0;
}
