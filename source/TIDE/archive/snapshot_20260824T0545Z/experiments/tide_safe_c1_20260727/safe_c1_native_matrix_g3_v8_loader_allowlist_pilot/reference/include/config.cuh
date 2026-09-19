// Config file of data type
// Created on 24-01-05

#pragma once
#define short float // Comment this out when working with integer files such as MPEG

// ================================================================
// C3: Query Scheduling & Load Balancing Configuration
// ================================================================

// Warp-level collaboration
#define C3_WARP_SIZE 32
#define C3_ENABLE_WARP_COLLAB 1  // 1 = enabled, 0 = disabled

// Query bucketing
#define C3_ENABLE_BUCKETING 1    // 1 = enabled, 0 = disabled
#define C3_MAX_BUCKETS 4         // Maximum number of buckets
#define C3_BUCKET_THRESHOLD_LOW 50   // Low difficulty threshold
#define C3_BUCKET_THRESHOLD_HIGH 200 // High difficulty threshold

// Dynamic block configuration
#define C3_MIN_THREADS_PER_BLOCK 128
#define C3_MAX_THREADS_PER_BLOCK 1024
#define C3_DEFAULT_THREADS 256    // Reduced from 512 for better occupancy