#include <iostream>
#include <string>
#include <vector>
#include "ocl.hpp"

using namespace std;

const int numLayers = 4;
const int numLevels = numLayers - 1;
const int maxBucketSize = 64;

struct Fargs {
  float ptMin;
  float regionOriginX;
  float regionOriginY;
  float regionOriginRadius;
  float thetaCut;
  float phiCut;
};

struct XYZVector {
  vector<float> X;
  vector<float> Y;
  vector<float> Z;
};

void dataLoad(string filename, Fargs args, vector<vector<int>>& innerPointIds,
              vector<vector<int>>& outerPointIds, vector<XYZVector>& points) {
  FILE* fin = fopen(filename.c_str(), "r");

  fscanf(fin, "%f", &args.ptMin);
  fscanf(fin, "%f", &args.regionOriginX);
  fscanf(fin, "%f", &args.regionOriginY);
  fscanf(fin, "%f", &args.regionOriginRadius);
  fscanf(fin, "%f", &args.thetaCut);
  fscanf(fin, "%f", &args.phiCut);

  for (int level = 0; level < numLevels; level++) {
    int doubletCount = 0;
    fscanf(fin, "%d", &doubletCount);
    innerPointIds[level].resize(doubletCount);
    outerPointIds[level].resize(doubletCount);
    for (int j = 0; j < doubletCount; j++) {
      fscanf(fin, "%d", &innerPointIds[level][j]);
      fscanf(fin, "%d", &outerPointIds[level][j]);
    }

    for (int k = 0; k < 2; k++) {
      int pointCount = 0;
      fscanf(fin, "%d", &pointCount);

      for (int j = 0; j < pointCount; j++) {
        float x, y, z;
        fscanf(fin, "%f", &x);
        fscanf(fin, "%f", &y);
        fscanf(fin, "%f", &z);
        if (k == 0) {
          points[level].X.push_back(x);
          points[level].Y.push_back(x);
          points[level].Z.push_back(x);
        }
        if (level == numLevels - 1 && k == 1) {
          points[level + 1].X.push_back(x);
          points[level + 1].Y.push_back(x);
          points[level + 1].Z.push_back(x);
        }
      }
    }
  }

  fclose(fin);
}

int main(int argc, char** argv) {
  OCL ocl(0);
  cl_kernel initBuckets_kernel =
      ocl.buildKernel("kernels.cl", "initBuckets",
                      "-D MAX_BUCKET_SIZE=" + to_string(maxBucketSize));
  cl_kernel connectDoublets_kernel =
      ocl.buildKernel("kernels.cl", "connectDoublets",
                      "-D MAX_BUCKET_SIZE=" + to_string(maxBucketSize));

  Fargs args;
  vector<vector<int>> innerPointIds(numLevels);
  vector<vector<int>> outerPointIds(numLevels);
  vector<XYZVector> points(numLayers);

  dataLoad("log.in", args, innerPointIds, outerPointIds, points);

  vector<cl_mem> d_innerPointIds(numLevels);
  vector<cl_mem> d_outerPointIds(numLevels);
  int doubletStarts[numLayers];
  doubletStarts[0] = 0;

  for (int level = 0; level < numLevels; level++) {
    d_innerPointIds[level] = ocl.createAndUpload(innerPointIds[level]);
    d_outerPointIds[level] = ocl.createAndUpload(outerPointIds[level]);
    doubletStarts[level + 1] =
        doubletStarts[level] + innerPointIds[level].size();
  }
  int totalPointCount = 0;
  for (int layer = 0; layer < numLayers; layer++) {
    totalPointCount += points[layer].X.size();
  }

  vector<int> pointBuckets(totalPointCount * maxBucketSize);
  vector<int> pointBucketSizes(totalPointCount * maxBucketSize, 0);

  cl_mem d_pointBuckets = ocl.createAndUpload(pointBuckets);
  cl_mem d_pointBucketsSizes = ocl.createAndUpload(pointBucketSizes);

  for (int level = 0; level < numLevels; level++) {
    ocl.execute(initBuckets_kernel, 1, {1024}, {64}, d_innerPointIds[level],
                (int)innerPointIds[level].size(), d_pointBuckets,
                d_pointBucketsSizes);
  }

  vector<int> connectedDoublets(doubletStarts[numLayers - 1] * maxBucketSize);
  vector<int> connectedDoubletsSizes(doubletStarts[numLayers - 1], 0);

  cl_mem d_connectedDoublets = ocl.createAndUpload(connectedDoublets);
  cl_mem d_connectedDoubletsSizes = ocl.createAndUpload(connectedDoubletsSizes);

  for (int level = 0; level < numLevels; level++) {
    ocl.execute(connectDoublets_kernel, 1, {1024}, {64}, d_outerPointIds[level],
                doubletStarts[level], doubletStarts[level + 1], d_pointBuckets,
                d_pointBucketsSizes, d_connectedDoublets,
                d_connectedDoubletsSizes);
  }

  auto result = ocl.download<int>(d_connectedDoublets);
  for (int i = 0; i < 150; i++) {
    cout << i << " " << outerPointIds[0][i] << ": ";
    for (int n = 0; n < maxBucketSize; n++) {
      cout << result[i * maxBucketSize + n] << " ";
    }
    cout << "\n";
  }
}
