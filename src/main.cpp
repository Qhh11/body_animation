#include <iostream>
#include <string>

#include "smpl_model.h"
#include "logging.h"          
#include "amass_runners.h"
#include "openpose_runners.h"

int main(int argc, char** argv) {
    google::InitGoogleLogging("SMPL_Optimizer");
    std::cout << ">>> Start SMPL Project...\n";

    SmplModel model;
    model.LoadFromNPZ("../../Data/models/smpl_neutral.npz");

    GMMPosePrior gmm_prior;
    gmm_prior.LoadFromNPZ("../../Data/models/gmm_08.npz");

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

    // default: run OpenPose sequence
    app::RunOpenPoseSequenceInference(model, &gmm_prior);
    return 0;
}
