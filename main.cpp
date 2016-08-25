#include <cmath>
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

struct d_XYZVector {
  cl_mem X;
  cl_mem Y;
  cl_mem Z;
};

d_XYZVector createAndUploadXYZVector(OCL& ocl, XYZVector vec) {
  d_XYZVector result;
  result.X = ocl.createAndUpload(vec.X);
  result.Y = ocl.createAndUpload(vec.Y);
  result.Z = ocl.createAndUpload(vec.Z);
  return result;
}

void dataLoad(string filename, Fargs& args, vector<vector<int>>& innerPointIds,
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
                      " -D TUPLET_SIZE=" + to_string(numLayers) +
                          " -D MAX_BUCKET_SIZE=" + to_string(maxBucketSize));
  cl_kernel connectDoublets_kernel =
      ocl.buildKernel("kernels.cl", "connectDoublets",
                      " -D TUPLET_SIZE=" + to_string(numLayers) +
                          " -D MAX_BUCKET_SIZE=" + to_string(maxBucketSize));
  cl_kernel findNTupletsFirstLevel_kernel =
      ocl.buildKernel("kernels.cl", "findNTupletsFirstLevel",
                      "-D TUPLET_SIZE=" + to_string(numLayers) +
                          " -D MAX_BUCKET_SIZE=" + to_string(maxBucketSize));
  cl_kernel findNTuplets_kernel =
      ocl.buildKernel("kernels.cl", "findNTuplets",
                      "-D TUPLET_SIZE=" + to_string(numLayers) +
                          " -D MAX_BUCKET_SIZE=" + to_string(maxBucketSize));

  Fargs args;
  vector<vector<int>> innerPointIds(numLevels);
  vector<vector<int>> outerPointIds(numLevels);
  vector<XYZVector> points(numLayers);
  vector<d_XYZVector> d_points(numLayers);

  dataLoad("log.in", args, innerPointIds, outerPointIds, points);

  for (int layer = 0; layer < numLayers; layer++) {
    d_points[layer] = createAndUploadXYZVector(ocl, points[layer]);
  }

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

  vector<vector<int>> pointBuckets(numLevels);
  vector<vector<int>> pointBucketsSizes(numLevels);

  vector<cl_mem> d_pointBuckets(numLevels);
  vector<cl_mem> d_pointBucketsSizes(numLevels);

  for (int level = 0; level < numLevels; level++) {
    pointBuckets[level].resize(maxBucketSize * points[level + 1].X.size());
    pointBucketsSizes[level].resize(points[level + 1].X.size());
    d_pointBuckets[level] = ocl.createAndUpload(pointBuckets[level]);
    d_pointBucketsSizes[level] = ocl.createAndUpload(pointBucketsSizes[level]);
    ocl.execute(initBuckets_kernel, 1, {1}, {1}, d_outerPointIds[level],
                (int)outerPointIds[level].size(), d_pointBuckets[level],
                d_pointBucketsSizes[level]);
  }

  vector<int> connectedDoublets(doubletStarts[numLayers - 1] * maxBucketSize);
  vector<int> connectedDoubletsSizes(doubletStarts[numLayers - 1], 0);

  cl_mem d_connectedDoublets = ocl.createAndUpload(connectedDoublets);
  cl_mem d_connectedDoubletsSizes = ocl.createAndUpload(connectedDoubletsSizes);

  for (int level = 1; level < numLevels; level++) {
    ocl.execute(
        connectDoublets_kernel, 1, {1}, {1}, d_innerPointIds[level - 1],
        d_innerPointIds[level], d_outerPointIds[level], doubletStarts[level],
        doubletStarts[level + 1], d_pointBuckets[level - 1],
        d_pointBucketsSizes[level - 1], d_connectedDoublets,
        d_connectedDoubletsSizes, d_points[level - 1].X, d_points[level - 1].Y,
        d_points[level - 1].Z, d_points[level + 0].X, d_points[level + 0].Y,
        d_points[level + 0].Z, d_points[level + 1].X, d_points[level + 1].Y,
        d_points[level + 1].Z, args.ptMin, args.regionOriginX,
        args.regionOriginY, args.regionOriginRadius, args.thetaCut,
        args.phiCut);
  }

  int maxNTupletCount =
      innerPointIds[numLevels - 1].size() * pow(maxBucketSize, numLevels - 1);
  vector<int> nTuplets(maxNTupletCount * (numLayers + 1));
  vector<int> nTupletCount(1);

  cl_mem d_nTupletsA = ocl.createAndUpload(nTuplets);
  cl_mem d_nTupletsB = ocl.createAndUpload(nTuplets);
  cl_mem d_nTupletCountA = ocl.createAndUpload(nTupletCount);
  cl_mem d_nTupletCountB = ocl.createAndUpload(nTupletCount);

  ocl.execute(findNTupletsFirstLevel_kernel, 1, {1}, {1},
              doubletStarts[numLayers - 2], doubletStarts[numLayers - 1],
              d_connectedDoublets, d_connectedDoubletsSizes, d_nTupletsA,
              d_nTupletCountA, d_outerPointIds[numLevels - 1],
              d_outerPointIds[numLevels - 2], d_innerPointIds[numLevels - 2]);

  for (int iteration = 1; iteration <= numLayers - 3; iteration++) {
    ocl.execute(findNTuplets_kernel, 1, {1}, {1},
                doubletStarts[numLayers - 2 - iteration], d_connectedDoublets,
                d_connectedDoubletsSizes, d_nTupletsA, d_nTupletCountA,
                d_nTupletsB, d_nTupletCountB,
                d_innerPointIds[numLayers - 3 - iteration], iteration);
    swap(d_nTupletsA, d_nTupletsB);
    swap(d_nTupletCountA, d_nTupletCountB);
    int pattern = 0;
    clEnqueueFillBuffer(ocl.queue, d_nTupletCountB, &pattern, sizeof(int), 0,
                        sizeof(int), 0, NULL, NULL);
  }

  ocl.downloadTo(d_nTupletCountA, nTupletCount);
  ocl.downloadTo(d_nTupletsA, nTuplets);

  ofstream file("log.out");
  for (int i = 0; i < nTupletCount[0]; i++) {
    for (int n = 0; n < numLayers; n++) {
      file << nTuplets[i * (numLayers + 1) + n + 1] << " ";
    }
    file << "\n";
  }
}
