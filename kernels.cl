// template parameters:  MAX_BUCKET_SIZE

__kernel void initBuckets(__global int* innerPointIds, int doubletCount,
                          __global int* pointBuckets,
                          __global int* bucketSizes) {
  int tidx = get_global_id(0);
  for (int doubletId = tidx; doubletId < doubletCount;
       doubletId += get_global_size(0)) {
    int pointId = innerPointIds[doubletId];
    int oldSize = atomic_inc(&bucketSizes[pointId]);
    if (oldSize < MAX_BUCKET_SIZE)
      pointBuckets[pointId * MAX_BUCKET_SIZE + oldSize] = doubletId;
  }
}

__kernel void connectDoublets() { int tidx = get_global_id(0); }
