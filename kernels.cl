// template parameters:  MAX_BUCKET_SIZE

__kernel void initBuckets(__global int* innerPointIds, int doubletCount,
                          __global int* pointBuckets,
                          __global int* pointBucketsSizes) {
  int tidx = get_global_id(0);
  for (int doubletId = tidx; doubletId < doubletCount;
       doubletId += get_global_size(0)) {
    int pointId = innerPointIds[doubletId];
    int oldSize = atomic_inc(&pointBucketsSizes[pointId]);
    if (oldSize < MAX_BUCKET_SIZE)
      pointBuckets[pointId * MAX_BUCKET_SIZE + oldSize] = doubletId;
  }
}

__kernel void connectDoublets(__global int* outerPointIds, int doubletIdStart,
                              int doubletIdEnd, __global int* pointBuckets,
                              __global int* pointBucketsSizes,
                              __global int* connectedDoublets,
                              __global int* connectedDoubletsSizes) {
  int tidx = get_global_id(0);

  for (int doubletId = tidx; doubletId < (doubletIdEnd - doubletIdStart);
       doubletId += get_global_size(0)) {
    for (int p = 0; p < pointBucketsSizes[outerPointIds[doubletId]]; p++) {
      int connectedDoubletId =
          pointBuckets[outerPointIds[doubletId] * MAX_BUCKET_SIZE + p];
      int oldSize = connectedDoubletsSizes[doubletIdStart + doubletId]++;
      if (oldSize < MAX_BUCKET_SIZE)
        connectedDoublets[(doubletIdStart + doubletId) * MAX_BUCKET_SIZE +
                          oldSize] = connectedDoubletId;
    }
  }
}
