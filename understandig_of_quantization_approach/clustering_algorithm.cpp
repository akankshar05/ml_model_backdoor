#include <iostream>
#include <vector>
#include <limits>

using namespace std;

const int matrixSize = 4;


float getDifference(float a, float b) {
    return abs(a - b);
}

void assignClusters(float matrix[matrixSize][matrixSize], float centers[4]) {
    vector<float> cluster[4];

    // Assign each value to the cluster with the minimum difference
    for (int i = 0; i < matrixSize; ++i) {
        for (int j = 0; j < matrixSize; ++j) {
            float minDiff = numeric_limits<float>::max();
            int clusterIndex = -1;

            // Find the cluster with the minimum difference
            for (int k = 0; k < 4; ++k) {
                float diff = getDifference(matrix[i][j], centers[k]);
                if (diff < minDiff) {
                    minDiff = diff;
                    clusterIndex = k;
                }
            }

            // Assign the value to the corresponding cluster
            cluster[clusterIndex].push_back(matrix[i][j]);
        }
    }

    // Display the clusters
    for (int k = 0; k < 4; ++k) {
        cout << "Cluster " << k + 1 << ": ";
        for (float value : cluster[k]) {
            cout << value << " ";
        }
        cout << endl;
    }
}

int main() {
    float matrix[matrixSize][matrixSize] = {
        {0.86, -1.88, -1.49, -0.51},
        {-1.7, 1.58, 0.12, -2.38},
        {1.9, -2.19, 1.37, -2.79},
        {1.92, 0.46, -2.85, -0.61}
    };

    // chosen centroids
    float centers[4] = {-2.85, -1.26, 0.33, 1.92};

    // Assign the remaining values to clusters
    assignClusters(matrix, centers);

    return 0;
}

