#include <cnpy.h>

using namespace std;
using namespace cnpy;
int main(int argc, char** argv) {
    string npz_path = "../../Data/star_1_1/neutral/model.npz";
    string out_png  = "out.png";
    npz_t npz = cnpy::npz_load(npz_path);
    for (auto& kv : npz) {
        const std::string& key = kv.first;
        const cnpy::NpyArray& arr = kv.second;
        std::cout << "  " << key << "  shape=(";
        for (size_t i = 0; i < arr.shape.size(); ++i) {
            std::cout << arr.shape[i];
            if (i + 1 < arr.shape.size()) std::cout << ",";
        }
        std::cout << ")  bytes=" << arr.word_size << "\n";
    }
}