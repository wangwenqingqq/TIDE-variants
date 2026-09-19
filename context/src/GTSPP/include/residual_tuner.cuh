// ================================================================
// Residual Pruning Tuner for GTS
// ================================================================
//
// Provides:
//   1. Offline Auto-Tuning: Learn per-level (alpha, beta, gamma) from pilot queries
//   2. Piecewise-Linear LUT fitting for enhanced inference
//   3. Online Adaptation: Adjust gamma based on runtime recall feedback
//
// The tuner collects training data by traversing the tree with pilot queries,
// computing actual distances to validate the residual between triangle
// inequality lower bounds and true distances.
//
// Safety guarantee: Uses conservative quantile regression (5th percentile)
// so the predicted residual is almost always <= actual residual.
//
// ================================================================

#ifndef RESIDUAL_TUNER_CUH
#define RESIDUAL_TUNER_CUH

#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <numeric>
#include <cstdio>
#include <cstring>
#include "config.cuh"
#include "tree.cuh"
#include "residual_pruning.cuh"

// A single training sample: (ratio, delta_true, level)
struct ResidualSample {
    float ratio;       // dis_lb / Rk (normalized lower bound)
    float delta_true;  // (d_true_min - dis_lb) / Rk (normalized true residual)
    int level;         // tree level where this sample was collected
};


class ResidualPruningTuner {
private:
    // ========= Tuning Hyperparameters =========
    static constexpr int NUM_PILOT_QUERIES = 512;
    static constexpr float SAFETY_QUANTILE = 0.05f;   // 5th percentile (leaf-only training provides extra safety)
    static constexpr int SAMPLE_NEIGHBORS = 2048;       // Number of random neighbors to estimate Rk
    
    // ========= Learned Parameters =========
    float alpha_[RP_MAX_LEVELS];
    float beta_[RP_MAX_LEVELS];
    float gamma_[RP_MAX_LEVELS];
    
    // LUT parameters
    float lut_breaks_[RP_LUT_SIZE];
    float lut_slopes_[RP_LUT_SIZE];
    float lut_intercepts_[RP_LUT_SIZE];
    int lut_num_segments_;
    
    int mode_;        // 1=linear, 2=lut
    int tree_h_;      // tree height
    
    // ========= Online Adaptation State =========
    float target_recall_;
    float current_gamma_scale_;
    int adaptation_step_;
    
    
    // ================================================================
    // CPU Distance Computation (for training data collection)
    // ================================================================
    static float compute_dist_cpu(const short* data, int idx1, int idx2, int dim, int metric_type) {
        float result = 0.0f;
        if (metric_type == 2) {  // L2
            for (int j = 0; j < dim; j++) {
                float diff = (float)data[idx1 * dim + j] - (float)data[idx2 * dim + j];
                result += diff * diff;
            }
            return sqrtf(result);
        } else if (metric_type == 1) {  // L1
            for (int j = 0; j < dim; j++) {
                result += fabsf((float)data[idx1 * dim + j] - (float)data[idx2 * dim + j]);
            }
            return result;
        } else if (metric_type == 0) {  // Linf
            for (int j = 0; j < dim; j++) {
                float diff = fabsf((float)data[idx1 * dim + j] - (float)data[idx2 * dim + j]);
                result = fmaxf(result, diff);
            }
            return result;
        } else if (metric_type == 5) {  // Cosine
            float sa1 = 0, sa2 = 0, sa3 = 0;
            for (int j = 0; j < dim; j++) {
                float a = (float)data[idx1 * dim + j];
                float b = (float)data[idx2 * dim + j];
                sa1 += a * a;
                sa2 += b * b;
                sa3 += a * b;
            }
            sa1 = sqrtf(sa1);
            sa2 = sqrtf(sa2);
            if (sa1 * sa2 < 1e-10f) return 0.0f;
            float cos_val = sa3 / (sa1 * sa2);
            if (cos_val > 1.0f) cos_val = 0.9999999f;
            return fabsf(acosf(cos_val) * 180.0f / 3.1415926f);
        } else {  // Default: L2
            for (int j = 0; j < dim; j++) {
                float diff = (float)data[idx1 * dim + j] - (float)data[idx2 * dim + j];
                result += diff * diff;
            }
            return sqrtf(result);
        }
    }
    
    
    // ================================================================
    // Estimate k-NN Search Radius (Rk) for a Query
    // ================================================================
    float estimateRk(int q_idx, const short* data, int n, int dim, int metric_type, int k) {
        std::vector<float> dists;
        dists.reserve(SAMPLE_NEIGHBORS);
        std::mt19937 rng(q_idx * 137 + 42);
        std::uniform_int_distribution<int> dist_gen(0, n - 1);
        
        for (int i = 0; i < SAMPLE_NEIGHBORS; ++i) {
            int r = dist_gen(rng);
            if (r != q_idx) {
                dists.push_back(compute_dist_cpu(data, q_idx, r, dim, metric_type));
            }
        }
        std::sort(dists.begin(), dists.end());
        int idx = std::min(k, (int)dists.size() - 1);
        return fmaxf(dists[idx], 1e-6f);
    }
    
    
    // ================================================================
    // Collect Training Samples: Traverse Tree Per Level
    // ================================================================
    void collectSamples(
        std::vector<std::vector<ResidualSample>>& level_samples,
        TN* node_list, int num_nodes, const short* data,
        int* id_list, int dim, int n, int metric_type)
    {
        std::mt19937 rng(2024);
        std::uniform_int_distribution<int> query_dist(0, n - 1);
        
        // Select pilot queries
        std::vector<int> pilot_queries;
        for (int i = 0; i < NUM_PILOT_QUERIES; ++i) {
            pilot_queries.push_back(query_dist(rng));
        }
        
        printf("[ResidualTuner] Collecting samples from %d pilot queries...\n",
               (int)pilot_queries.size());
        
        // Estimate Rk for each pilot query
        std::vector<float> Rks;
        for (int q : pilot_queries) {
            float rk = estimateRk(q, data, n, dim, metric_type, 10);
            Rks.push_back(rk);
        }
        
        // Rk statistics
        std::vector<float> rk_sorted = Rks;
        std::sort(rk_sorted.begin(), rk_sorted.end());
        float rk_avg = 0;
        for (float v : Rks) rk_avg += v;
        rk_avg /= Rks.size();
        printf("  Rk stats: avg=%.4f, p25=%.4f, p50=%.4f, p75=%.4f\n",
               rk_avg,
               rk_sorted[rk_sorted.size() / 4],
               rk_sorted[rk_sorted.size() / 2],
               rk_sorted[rk_sorted.size() * 3 / 4]);
        
        // Traverse tree level by level
        int level_start = 1;  // Level 1 starts at node index 1
        
        for (int level = 1; level < tree_h_; ++level) {
            int nnum = 1;
            for (int i = 0; i < level; i++) nnum *= TREE_ORDER;
            
            // Sample up to 500 nodes at this level
            int node_step = std::max(1, nnum / 500);
            // Sample up to 100 queries per level
            int query_step = std::max(1, (int)pilot_queries.size() / 100);
            
            for (size_t qi = 0; qi < pilot_queries.size(); qi += query_step) {
                int q = pilot_queries[qi];
                float Rk = Rks[qi];
                
                for (int ni = 0; ni < nnum; ni += node_step) {
                    int nid = level_start + ni;
                    if (nid >= num_nodes) break;
                    if (node_list[nid].pid < 0) continue;
                    
                    TN node = node_list[nid];
                    
                    // CRITICAL FIX: Only use leaf nodes where d_min is exact.
                    // Internal nodes used d_qp as d_min estimate, which is almost
                    // always LARGER than the true subtree minimum, causing delta_true
                    // to be severely overestimated → too aggressive pruning.
                    if (!node.is_leaf) continue;
                    
                    float d_qp = compute_dist_cpu(data, q, node.pid, dim, metric_type);
                    
                    // Triangle inequality lower bound
                    float dis_lb = fmaxf(0.0f, d_qp - node.min_dis);
                    
                    // Bidirectional tightening
                    if (nid % TREE_ORDER != 0 && (nid + 1) < num_nodes) {
                        float dis_lb2 = d_qp - node_list[nid + 1].min_dis;
                        dis_lb = fmaxf(dis_lb, dis_lb2);
                    }
                    
                    // Compute actual minimum distance to data in subtree
                    float d_min = 1e30f;
                    if (node.is_leaf && node.size > 0) {
                        // For leaf nodes: exact d_min
                        for (int di = 0; di < node.size; ++di) {
                            int data_id = id_list[node.lid + di];
                            float d = compute_dist_cpu(data, q, data_id, dim, metric_type);
                            d_min = fminf(d_min, d);
                        }
                    }
                    // Note: Internal nodes are skipped (only leaf samples used for safety)
                    
                    if (d_min > 1e29f) continue;
                    
                    // The true residual: how much tighter we can make the LB
                    float delta_true = fmaxf(0.0f, d_min - dis_lb);
                    float ratio = dis_lb / fmaxf(Rk, 1e-8f);
                    
                    // Filter extreme outliers
                    if (ratio < 10.0f) {
                        ResidualSample s;
                        s.ratio = ratio;
                        s.delta_true = delta_true / fmaxf(Rk, 1e-8f);  // Normalize
                        s.level = level;
                        level_samples[level].push_back(s);
                    }
                }
            }
            
            level_start += nnum;
        }
    }
    
    
    // ================================================================
    // Fit Conservative Linear Model per Level
    // Uses quantile regression at SAFETY_QUANTILE for safety
    // ================================================================
    void fitLinearModel(const std::vector<ResidualSample>& samples,
                        float& alpha, float& beta) {
        if (samples.size() < 10) {
            alpha = 0.0f;
            beta = 0.0f;
            return;
        }
        
        // Bin samples by ratio for robust fitting
        const int NUM_BINS = 20;
        std::vector<std::vector<float>> bins(NUM_BINS);
        
        float max_ratio = 0.0f;
        for (const auto& s : samples) {
            max_ratio = fmaxf(max_ratio, s.ratio);
        }
        if (max_ratio < 1e-6f) max_ratio = 1.0f;
        
        for (const auto& s : samples) {
            int bin = std::min((int)(s.ratio / max_ratio * NUM_BINS), NUM_BINS - 1);
            bins[bin].push_back(s.delta_true);
        }
        
        // Get conservative (5th percentile) delta for each bin
        std::vector<float> bin_ratios, bin_deltas;
        for (int i = 0; i < NUM_BINS; ++i) {
            if (bins[i].size() < 5) continue;
            std::sort(bins[i].begin(), bins[i].end());
            int idx = std::max(0, (int)(bins[i].size() * SAFETY_QUANTILE));
            float ratio_mid = (i + 0.5f) / NUM_BINS * max_ratio;
            bin_ratios.push_back(ratio_mid);
            bin_deltas.push_back(bins[i][idx]);
        }
        
        if (bin_ratios.size() < 2) {
            alpha = 0.0f;
            beta = 0.0f;
            return;
        }
        
        // Least squares fit: delta = alpha * ratio + beta
        float sum_r = 0, sum_d = 0, sum_rr = 0, sum_rd = 0;
        int n = bin_ratios.size();
        for (int i = 0; i < n; ++i) {
            sum_r += bin_ratios[i];
            sum_d += bin_deltas[i];
            sum_rr += bin_ratios[i] * bin_ratios[i];
            sum_rd += bin_ratios[i] * bin_deltas[i];
        }
        
        float denom = n * sum_rr - sum_r * sum_r;
        if (fabsf(denom) < 1e-12f) {
            alpha = 0.0f;
            beta = (n > 0) ? sum_d / n : 0.0f;
        } else {
            alpha = (n * sum_rd - sum_r * sum_d) / denom;
            beta = (sum_d - alpha * sum_r) / n;
        }
        
        // Apply safety margin: reduce predictions by 20%
        alpha *= 0.8f;
        beta *= 0.8f;
    }
    
    
    // ================================================================
    // Fit Piecewise-Linear LUT (global across all levels)
    // ================================================================
    void fitPiecewiseLUT(const std::vector<ResidualSample>& all_samples) {
        if (all_samples.size() < 20) {
            lut_num_segments_ = 1;
            lut_breaks_[0] = 0.0f;
            lut_slopes_[0] = 0.0f;
            lut_intercepts_[0] = 0.0f;
            return;
        }
        
        // Sort all samples by ratio
        std::vector<ResidualSample> sorted = all_samples;
        std::sort(sorted.begin(), sorted.end(),
                  [](const ResidualSample& a, const ResidualSample& b) {
                      return a.ratio < b.ratio;
                  });
        
        // Determine number of segments
        int target_segments = std::min(8, (int)sorted.size() / 20);
        target_segments = std::max(target_segments, 2);
        lut_num_segments_ = target_segments;
        
        int per_seg = sorted.size() / target_segments;
        
        for (int seg = 0; seg < target_segments; ++seg) {
            int start = seg * per_seg;
            int end = (seg == target_segments - 1) ? (int)sorted.size() : (seg + 1) * per_seg;
            
            lut_breaks_[seg] = sorted[start].ratio;
            
            // Use QUANTILE-based fitting per segment (consistent with per-level model)
            // This is more conservative than least squares
            std::vector<float> seg_deltas;
            seg_deltas.reserve(end - start);
            for (int i = start; i < end; ++i) {
                seg_deltas.push_back(sorted[i].delta_true);
            }
            std::sort(seg_deltas.begin(), seg_deltas.end());
            
            int q_idx = std::max(0, (int)(seg_deltas.size() * SAFETY_QUANTILE));
            float q_delta = seg_deltas[q_idx];
            
            // Conservative constant per segment
            lut_slopes_[seg] = 0.0f;
            lut_intercepts_[seg] = q_delta * 0.8f;  // Extra 20% safety margin
        }
        
        // Fill remaining slots
        for (int seg = target_segments; seg < RP_LUT_SIZE; ++seg) {
            lut_breaks_[seg] = 1e10f;
            lut_slopes_[seg] = 0.0f;
            lut_intercepts_[seg] = 0.0f;
        }
    }
    
    
    // ================================================================
    // Upload Parameters to GPU Constant Memory
    // ================================================================
    void uploadToGPU() {
        upload_rp_constants(alpha_, beta_, gamma_, RP_MAX_LEVELS,
                           lut_breaks_, lut_slopes_, lut_intercepts_, lut_num_segments_,
                           mode_);
    }
    

public:
    // ================================================================
    // Constructor
    // ================================================================
    ResidualPruningTuner()
        : mode_(1), tree_h_(3), target_recall_(0.999f),
          current_gamma_scale_(1.0f), adaptation_step_(0)
    {
        memset(alpha_, 0, sizeof(alpha_));
        memset(beta_, 0, sizeof(beta_));
        memset(gamma_, 0, sizeof(gamma_));
        memset(lut_breaks_, 0, sizeof(lut_breaks_));
        memset(lut_slopes_, 0, sizeof(lut_slopes_));
        memset(lut_intercepts_, 0, sizeof(lut_intercepts_));
        lut_num_segments_ = 0;
    }
    
    
    // ================================================================
    // [Contribution 1+2+3] Initialize for Scale-Based Pruning
    // No offline training needed - calibration directly finds optimal scales
    // ================================================================
    void AutoTuneAndUpload(
        TN* node_list, int num_nodes, short* data_h,
        int* id_list_h, int dim, int n, int metric_type, int tree_h)
    {
        tree_h_ = tree_h;
        mode_ = 1;  // Scale-based pruning enabled
        
        printf("\n================================================================\n");
        printf("[ResidualPruningTuner] Scale-Based Pruning Initialization\n");
        printf("================================================================\n");
        printf("  Tree height: %d\n", tree_h);
        printf("  Strategy: Per-Level Scale Calibration\n");
        printf("  Inference: 3 FLOPs per node (1 sub + 1 mul + 1 max)\n");
        printf("  vs MLP: ~40 FLOPs per node (10x fewer)\n");
        printf("  Calibration: Binary search on actual search recall\n");
        printf("================================================================\n");
        
        // Initialize: all scales = 1.0 (baseline, no extra pruning)
        // Levels 1-2: always 1.0 (handled by labelCNode)
        // Levels 3+: will be calibrated per-level
        for (int level = 0; level < RP_MAX_LEVELS; ++level) {
            gamma_[level] = 1.0f;   // scale = 1.0 means baseline
            alpha_[level] = 0.0f;
            beta_[level] = 0.0f;
        }
        lut_num_segments_ = 0;
        
        printf("\n  Initial state: all scales = 1.0 (baseline)\n");
        printf("  Awaiting per-level calibration...\n");
        printf("================================================================\n\n");
        
        // Upload baseline state to GPU
        uploadToGPU();
    }
    
    
    // ================================================================
    // [Contribution 4] Online Adaptation
    // Adjusts pruning aggressiveness based on measured recall
    // Call this after each search batch
    // ================================================================
    void OnlineAdapt(float measured_recall, float target_recall = 0.999f) {
        target_recall_ = target_recall;
        adaptation_step_++;
        
        printf("[OnlineAdapt] Step %d: Recall=%.4f (target=%.4f)\n",
               adaptation_step_, measured_recall, target_recall);
        
        if (measured_recall < target_recall) {
            // Recall too low → reduce scale excess toward baseline (1.0)
            for (int i = 0; i < RP_MAX_LEVELS; ++i) {
                gamma_[i] = 1.0f + (gamma_[i] - 1.0f) * 0.7f;
            }
            printf("  -> Reducing scales toward baseline\n");
            uploadToGPU();
        } else if (measured_recall > target_recall + 0.005f) {
            // Recall comfortably above target → increase scale for more speed
            for (int i = 0; i < RP_MAX_LEVELS; ++i) {
                gamma_[i] = 1.0f + (gamma_[i] - 1.0f) * 1.15f;
            }
            printf("  -> Increasing scales for more pruning\n");
            uploadToGPU();
        } else {
            printf("  -> Recall in target range, no adjustment needed.\n");
        }
    }
    
    
    // ================================================================
    // Online Adaptation without Ground Truth
    // Uses distance ratio as proxy for recall quality
    // ================================================================
    void OnlineAdaptFromDistances(float* prev_Rk, float* curr_Rk, int qnum) {
        // Compare search radius changes between batches
        // If Rk is getting smaller, pruning is working well
        // If Rk is getting larger, pruning might be too aggressive
        
        if (prev_Rk == nullptr || curr_Rk == nullptr || qnum == 0) return;
        
        float ratio_sum = 0.0f;
        int valid = 0;
        for (int i = 0; i < qnum; ++i) {
            if (prev_Rk[i] > 1e-8f && curr_Rk[i] > 1e-8f) {
                ratio_sum += curr_Rk[i] / prev_Rk[i];
                valid++;
            }
        }
        
        if (valid > 0) {
            float avg_ratio = ratio_sum / valid;
            // If avg_ratio is close to 1.0, recall is stable
            float estimated_recall = fminf(1.0f, 1.0f / fmaxf(avg_ratio, 0.5f));
            OnlineAdapt(estimated_recall);
        }
    }
    
    
    // ================================================================
    // Disable Learned Pruning (fallback to baseline)
    // ================================================================
    void DisablePruning() {
        int mode_off = 0;
        cudaMemcpyToSymbol(c_rp_mode, &mode_off, sizeof(int));
        printf("[ResidualPruningTuner] Learned pruning disabled (baseline mode).\n");
    }
    
    
    // ================================================================
    // Enable Learned Pruning
    // ================================================================
    void EnablePruning() {
        cudaMemcpyToSymbol(c_rp_mode, &mode_, sizeof(int));
        printf("[ResidualPruningTuner] Learned pruning enabled (mode=%d).\n", mode_);
    }
    
    
    // ================================================================
    // Set Scale for a Specific Level (for per-level calibration)
    // gamma_[level] stores the scale factor (>= 1.0)
    // ================================================================
    void SetLevelScale(int level, float scale) {
        if (level >= 0 && level < RP_MAX_LEVELS) {
            gamma_[level] = fmaxf(1.0f, scale);
        }
        uploadToGPU();
    }
    
    
    // ================================================================
    // Get current scale for a level
    // ================================================================
    float GetLevelScale(int level) const {
        if (level >= 0 && level < RP_MAX_LEVELS) return gamma_[level];
        return 1.0f;
    }
    
    
    // ================================================================
    // Set Gamma Scale (for calibration binary search)
    // ================================================================
    void SetGammaScale(float scale) {
        current_gamma_scale_ = scale;
        float adjusted_gamma[RP_MAX_LEVELS];
        for (int i = 0; i < RP_MAX_LEVELS; ++i) {
            adjusted_gamma[i] = gamma_[i] * current_gamma_scale_;
        }
        upload_rp_constants(alpha_, beta_, adjusted_gamma, RP_MAX_LEVELS,
                           lut_breaks_, lut_slopes_, lut_intercepts_, lut_num_segments_,
                           mode_);
    }
    
    
    // ================================================================
    // Print Current Status
    // ================================================================
    void PrintStatus() const {
        printf("\n[ResidualPruningTuner Status]\n");
        printf("  Mode: %s\n", (mode_ == 0) ? "Baseline (no pruning)" : "Scale-Based Pruning");
        printf("  Target recall: %.4f\n", target_recall_);
        for (int l = 1; l < tree_h_; ++l) {
            printf("  Level %d: scale=%.4f %s\n",
                   l, gamma_[l],
                   gamma_[l] > 1.001f ? "(active)" : "(baseline)");
        }
        printf("\n");
    }
    
    
    // ================================================================
    // Getters for external use
    // ================================================================
    int GetMode() const { return mode_; }
    float GetGammaScale() const { return current_gamma_scale_; }
    int GetTreeHeight() const { return tree_h_; }
};

#endif // RESIDUAL_TUNER_CUH
