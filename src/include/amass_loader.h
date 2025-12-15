#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <Eigen/Dense>

struct AmassFrame {
    Eigen::VectorXd beta;  // [10]
    Eigen::VectorXd theta; // [72]
    Eigen::Vector3d trans; // [3]
};

class AmassLoader {
public:
    static bool Load(const std::string& path, std::vector<AmassFrame>& out_frames) {
        std::cout << "[Loader] Reading formatted text file: " << path << std::endl;
        
        std::ifstream infile(path);
        if (!infile.is_open()) {
            std::cerr << "[Loader] Error: Could not open file. Did you run convert_amass.py?" << std::endl;
            return false;
        }

        int num_frames;
        infile >> num_frames;

        if (num_frames <= 0) {
            std::cerr << "[Loader] Error: Invalid number of frames: " << num_frames << std::endl;
            return false;
        }

        out_frames.clear();
        out_frames.reserve(num_frames);

        for (int i = 0; i < num_frames; ++i) {
            AmassFrame frame;
            frame.beta = Eigen::VectorXd(10);
            frame.theta = Eigen::VectorXd(72);
            
            // 1. Read Beta [10]
            for(int j=0; j<10; ++j) infile >> frame.beta[j];
            
            // 2. Read Theta [72]
            for(int j=0; j<72; ++j) infile >> frame.theta[j];
            
            // 3. Read Trans [3]
            infile >> frame.trans[0] >> frame.trans[1] >> frame.trans[2];

            out_frames.push_back(frame);
        }

        std::cout << "[Loader] Successfully loaded " << out_frames.size() << " frames." << std::endl;
        return true;
    }
};