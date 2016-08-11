// template parameters:  MAX_BUCKET_SIZE

__kernel initBuckets(int* innerPointIds, int doubletCount, int* doubletBuckets,
                     int* bucketSizes) {
  int tidx = get_global_id(0);
  for (int doubletId = 0; doubletId < idCount;
       doubletId += get_global_size(0)) {
    int pointId = innerPointIds[doubletId];
    int oldSize = atomic_inc(&bucketSizes[pointId]);
    doubletBuckets[oldSize] = id;
  }
}
