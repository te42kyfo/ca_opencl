// template parameters:  MAX_BUCKET_SIZE
__kernel void initBuckets(__global int* outerPointIds, int doubletCount,
                          __global int* pointBuckets,
                          __global int* pointBucketsSizes) {
  int tidx = get_global_id(0);
  for (int doubletId = tidx; doubletId < doubletCount;
       doubletId += get_global_size(0)) {
    int pointId = outerPointIds[doubletId];
    int oldSize = atomic_inc(&pointBucketsSizes[pointId]);
    if (oldSize < MAX_BUCKET_SIZE)
      pointBuckets[pointId * MAX_BUCKET_SIZE + oldSize] = doubletId;
  }
}

bool are_aligned_RZ(float r1, float z1, float r2, float z2, float r3, float z3,
                    float ptmin, float thetaCut) {
  float distance_13_squared = (r1 - r3) * (r1 - r3) + (z1 - z3) * (z1 - z3);
  float tan_12_13_half =
      fabs(z1 * (r2 - r3) + z2 * (r3 - r1) + z3 * (r1 - r2)) /
      distance_13_squared;

  return tan_12_13_half * ptmin <= thetaCut;
}

bool have_similar_curvature(float x1, float y1, float x2, float y2, float x3,
                            float y3, const float region_origin_x,
                            const float region_origin_y,
                            const float region_origin_radius,
                            const float phiCut) {
  float precision = 0.5f;
  float offset = x2 * x2 + y2 * y2;

  float bc = (x1 * x1 + y1 * y1 - offset) / 2.f;

  float cd = (offset - x3 * x3 - y3 * y3) / 2.f;

  float det = (x1 - x2) * (y2 - y3) - (x2 - x3) * (y1 - y2);

  // points are aligned
  if (fabs(det) < precision) return 1;

  float idet = 1.f / det;

  float x_center = (bc * (y2 - y3) - cd * (y1 - y2)) * idet;
  float y_center = (cd * (x1 - x2) - bc * (x2 - x3)) * idet;

  float radius = sqrt((x2 - x_center) * (x2 - x_center) +
                      (y2 - y_center) * (y2 - y_center));
  float centers_distance_squared =
      (x_center - region_origin_x) * (x_center - region_origin_x) +
      (y_center - region_origin_y) * (y_center - region_origin_y);

  float minimumOfIntesectionRange =
      (radius - region_origin_radius) * (radius - region_origin_radius) -
      phiCut;

  if (centers_distance_squared >= minimumOfIntesectionRange) {
    float maximumOfIntesectionRange =
        (radius + region_origin_radius) * (radius + region_origin_radius) +
        phiCut;
    return centers_distance_squared <= maximumOfIntesectionRange;
  } else {
    return false;
  }
}

// template parameters:  MAX_BUCKET_SIZE
__kernel void connectDoublets(
    global int* pointIds1, global int* pointIds2, global int* pointIds3,
    int doubletIdStart, int doubletIdEnd, global int* pointBuckets,
    global int* pointBucketsSizes, global int* connectedDoublets,
    global int* connectedDoubletsSizes, global float* pointsX1,
    global float* pointsY1, global float* pointsZ1, global float* pointsX2,
    global float* pointsY2, global float* pointsZ2, global float* pointsX3,
    global float* pointsY3, global float* pointsZ3, float ptMin,
    float regionOriginX, float regionOriginY, float regionOriginRadius,
    float thetaCut, float phiCut) {
  int tidx = get_global_id(0);

  for (int doubletId = tidx; doubletId < (doubletIdEnd - doubletIdStart);
       doubletId += get_global_size(0)) {
    for (int p = 0; p < pointBucketsSizes[pointIds2[doubletId]]; p++) {
      int connectedDoubletId =
          pointBuckets[pointIds2[doubletId] * MAX_BUCKET_SIZE + p];

      float x1 = pointsX1[pointIds1[connectedDoubletId]];
      float y1 = pointsY1[pointIds1[connectedDoubletId]];
      float z1 = pointsZ1[pointIds1[connectedDoubletId]];
      float x2 = pointsX2[pointIds2[doubletId]];
      float y2 = pointsY2[pointIds2[doubletId]];
      float z2 = pointsZ2[pointIds2[doubletId]];
      float x3 = pointsX3[pointIds3[doubletId]];
      float y3 = pointsY3[pointIds3[doubletId]];
      float z3 = pointsZ3[pointIds3[doubletId]];
      float r1 = hypot(x1, y1);
      float r2 = hypot(x2, y2);
      float r3 = hypot(x3, y3);

      if (are_aligned_RZ(r1, z1, r2, z2, r3, z3, ptMin, thetaCut) &&
          have_similar_curvature(x1, y1, x2, y2, x3, y3, regionOriginX,
                                 regionOriginY, regionOriginRadius, phiCut)) {
        int oldSize = connectedDoubletsSizes[doubletIdStart + doubletId]++;
        if (oldSize < MAX_BUCKET_SIZE)
          connectedDoublets[(doubletIdStart + doubletId) * MAX_BUCKET_SIZE +
                            oldSize] = connectedDoubletId;
        // printf(".");
      } else {
        // printf("#\n");
      }
    }
  }
}

// Tuplet: [nextDoublet | p1 | p2 | p3 | p4]
// template parameters:  MAX_BUCKET_SIZE, TUPLET_SIZE
__kernel void findNTupletsFirstLevel(
    int doubletIdStart, int doubletIdEnd, global int* connectedDoublets,
    global int* connectedDoubletsSizes, global int* nTuplets,
    global int* nTupletCount, global int* outerPointIds,
    global int* nextOuterPointIds, global int* nextInnerPointIds) {
  int tidx = get_global_id(0);

  for (int doubletId = tidx; doubletId < (doubletIdEnd - doubletIdStart);
       doubletId += get_global_size(0)) {
    int threadStart = atomic_add(
        nTupletCount, connectedDoubletsSizes[doubletIdStart + doubletId]);
    for (int conIdx = 0;
         conIdx < connectedDoubletsSizes[doubletIdStart + doubletId];
         conIdx++) {
      int nextDoublet =
          connectedDoublets[(doubletId + doubletIdStart) * MAX_BUCKET_SIZE +
                            conIdx];
      nTuplets[(threadStart + conIdx) * (TUPLET_SIZE + 1) + 0] = nextDoublet;
      nTuplets[(threadStart + conIdx) * (TUPLET_SIZE + 1) + 1] =
          outerPointIds[doubletId];
      nTuplets[(threadStart + conIdx) * (TUPLET_SIZE + 1) + 2] =
          nextOuterPointIds[nextDoublet];
      nTuplets[(threadStart + conIdx) * (TUPLET_SIZE + 1) + 3] =
          nextInnerPointIds[nextDoublet];
    }
  }
}

// Tuplet: [nextDoublet | p1 | p2 | p3 | p4]
// template parameters:  MAX_BUCKET_SIZE, TUPLET_SIZE
kernel void findNTuplets(int doubletOffset, global int* connectedDoublets,
                         global int* connectedDoubletsSizes,
                         global int* nTupletsSrc,
                         global const int* tupletCountSrc,
                         global int* nTupletsDst, global int* tupletCountDst,
                         global int* innerPointIds, int iteration) {
  int tidx = get_global_id(0);

  for (int tupletId = tidx; tupletId < *tupletCountSrc;
       tupletId += get_global_size(0)) {
    int currentDoublet = nTupletsSrc[(TUPLET_SIZE + 1) * tupletId + 0];
    int connectedDoubletCount =
        connectedDoubletsSizes[doubletOffset + currentDoublet];

    int threadStart = atomic_add(tupletCountDst, connectedDoubletCount);
    for (int conIdx = 0; conIdx < connectedDoubletCount; conIdx++) {
      int nextDoublet =
          connectedDoublets[(currentDoublet + doubletOffset) * MAX_BUCKET_SIZE +
                            conIdx];
      nTupletsDst[(threadStart + conIdx) * (TUPLET_SIZE + 1) + 0] = nextDoublet;
      for (int i = 0; i < iteration + 2; i++) {
        nTupletsDst[(threadStart + conIdx) * (TUPLET_SIZE + 1) + 1 + i] =
            nTupletsSrc[tupletId * (TUPLET_SIZE + 1) + 1 + i];
      }
      nTupletsDst[(threadStart + conIdx) * (TUPLET_SIZE + 1) + 3 + iteration] =
          innerPointIds[nextDoublet];
    }
  }
}
