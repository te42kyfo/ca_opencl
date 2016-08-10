#include <iostream>
#include <string>
#include <vector>

using namespace std;

const int numLayers = 4;
const int numLevels = numLayers - 1;

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

void dataLoad(string filename, Fargs args, vector<vector<int>>& innerDoubletIds,
              vector<vector<int>>& outerDoubleIds, vector<XYZVector>& points) {
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
    innerDoubletIds[level].resize(doubletCount);
    outerDoubleIds[level].resize(doubletCount);
    for (int j = 0; j < doubletCount; j++) {
      fscanf(fin, "%d", &innerDoubletIds[level][j]);
      fscanf(fin, "%d", &outerDoubleIds[level][j]);
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

        if (j < 3) cout << x << " " << y << " " << z << " ";
      }
      cout << "\n";
    }
    cout << "\n";
  }

  fclose(fin);
}

int main(int argc, char** argv) {
  Fargs args;
  vector<vector<int>> innerDoubletIds(numLevels);
  vector<vector<int>> outerDoubletIds(numLevels);
  vector<XYZVector> points(numLayers);
  dataLoad("log.in", args, innerDoubletIds, outerDoubletIds, points);
}
