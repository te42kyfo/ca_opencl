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

/*
bool have_similar_curvature(const OMPCACell* this, const OMPCACell* otherCell,
                            const float region_origin_x,
                            const float region_origin_y,
                            const float region_origin_radius,
                            const float phiCut) {
  float x1 = get_inner_x(otherCell);
  float y1 = get_inner_y(otherCell);

  float x2 = get_inner_x(this);
  float y2 = get_inner_y(this);

  float x3 = get_outer_x(this);
  float y3 = get_outer_y(this);

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
*/
__kernel void connectDoublets(
    global int* pointIds1, global int* pointIds2, global int* pointIds3,
    int doubletIdStart, int doubletIdEnd, global int* pointBuckets,
    global int* pointBucketsSizes, global int* connectedDoublets,
    global int* connectedDoubletsSizes, global float* x1, global float* y1,
    global float* z1, global float* x2, global float* y2, global float* z2,
    global float* x3, global float* y3, global float* z3, float ptmin,
    float thetaCut) {
  int tidx = get_global_id(0);

  for (int doubletId = tidx; doubletId < (doubletIdEnd - doubletIdStart);
       doubletId += get_global_size(0)) {
    for (int p = 0; p < pointBucketsSizes[pointIds2[doubletId]]; p++) {
      int connectedDoubletId =
          pointBuckets[pointIds2[doubletId] * MAX_BUCKET_SIZE + p];

      float r1 = hypot(x1[pointIds1[connectedDoubletId]],
                       y1[pointIds1[connectedDoubletId]]);
      float r2 = hypot(x2[pointIds2[doubletId]], y2[pointIds2[doubletId]]);
      float r3 = hypot(x3[pointIds3[doubletId]], y3[pointIds3[doubletId]]);

      if (are_aligned_RZ(r1, z1[pointIds1[connectedDoubletId]], r2,
                         z2[pointIds2[doubletId]], r3, z3[pointIds3[doubletId]],
                         ptmin, thetaCut)) {
        int oldSize = connectedDoubletsSizes[doubletIdStart + doubletId]++;
        if (oldSize < MAX_BUCKET_SIZE)
          connectedDoublets[(doubletIdStart + doubletId) * MAX_BUCKET_SIZE +
                            oldSize] = connectedDoubletId;
      }
    }
  }
}
