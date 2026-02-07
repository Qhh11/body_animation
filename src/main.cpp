#include <iostream>
#include <string>

#include "smpl_model.h"
#include "logging.h"
#include "amass_runners.h"
#include "openpose_runners.h"

static bool HasFlag(int argc, char** argv, const std::string& flag) {
    for (int i = 1; i < argc; ++i) if (flag == argv[i]) return true;
    return false;
}

static std::string GetArgValue(int argc, char** argv,
                               const std::string& key,
                               const std::string& def) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (key == argv[i]) return std::string(argv[i + 1]);
    }
    return def;
}
static int GetArgInt(int argc, char** argv,
                     const std::string& key,
                     int def) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (key == argv[i]) {
            try { return std::stoi(argv[i + 1]); }
            catch (...) { return def; }
        }
    }
    return def;
}

int main(int argc, char** argv) {
    google::InitGoogleLogging("SMPL_Optimizer");
    std::cout << ">>> Start SMPL Project...\n";

    SmplModel model;
    model.LoadFromNPZ("../../Data/models/smpl_neutral.npz");

    GMMPosePrior gmm_prior;
    gmm_prior.LoadFromNPZ("../../Data/models/gmm_08.npz");

    if (HasFlag(argc, argv, "--openpose-seq")) {
        const std::string seq_name = GetArgValue(argc, argv, "--input_seq",
                                "courtyard_bodyScannerMotions_00");
        const std::string out_dir  = GetArgValue(argc, argv, "--output_dir",
                                "./seq_out_motion_2");
        const int start = GetArgInt(argc, argv, "--start", 70);
        const int end   = GetArgInt(argc, argv, "--end",   140);

        app::RunOpenPoseSequenceInference(model, &gmm_prior, seq_name, out_dir, start, end);
        return 0;
    }


    if (argc > 1) {
        std::string arg1 = argv[1];

        if (arg1 == "--batch") {
            app::RunBatchValidation(model, &gmm_prior);
            return 0;
        }

        if (arg1 == "--openpose") {
            app::RunOpenPoseInference(model, &gmm_prior);
            return 0;
        }

        if (arg1 == "--seq-val") {
            if (argc < 3) {
                std::cerr << "Usage: " << argv[0] << " --seq-val <pose_file.txt>\n";
                return 1;
            }
            app::RunAmassValidationSequence(model, &gmm_prior, argv[2]);
            return 0;
        }

        // default: single AMASS file path
        app::RunAMASSValidation(model, &gmm_prior, arg1, 100, true);
        return 0;
    }
    return 0;
}
