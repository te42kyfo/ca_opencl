// template parameters:  MAX_BUCKET_SIZE

__kernel void initBuckets(__global int* innerPointIds, int doubletCount,
                          __global int* pointBuckets,
                          __global int* bucketSizes) {
  int tidx = get_global_id(0);
  for (int doubletId = tidx; doubletId < doubletCount;
       doubletId += get_global_size(0)) {
    int pointId = innerPointIds[doubletId];
    if (pointId < 5) printf("%d %d\n", doubletId, pointId);
    int oldSize = atomic_inc(&bucketSizes[pointId]);
    pointBuckets[pointId * MAX_BUCKET_SIZE + oldSize] = doubletId;
  }
}
