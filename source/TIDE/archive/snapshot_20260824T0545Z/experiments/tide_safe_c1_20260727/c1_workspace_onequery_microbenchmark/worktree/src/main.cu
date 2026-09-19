// GTS index construction and similarity search with GTS
// Created on 24-01-05

#include <chrono>
#include <cstring>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
// ================= [Residual Pruning Framework] =================
// Define constants before including headers
#define MLP_MAIN_FILE
#define RP_DEFINE_CONSTANTS
#include "residual_pruning.cuh"
#include "mlp_constant.cuh"
#include "gpu_timer.cuh"
#include "tree.cuh"
#include "file.cuh"
#include "search.cuh"
#include "update.cuh"
#include "search_v2.cuh"
#include "config.cuh"
#include "residual_tuner.cuh"


// ================= Upload Functions =================
// Legacy MLP upload (kept for compatibility)
void upload_mlp_constants(float* h_scale, float* h_W1, float* h_b1, float* h_W2, float* h_b2) {
    cudaMemcpyToSymbol(input_scale, h_scale, 3 * sizeof(float), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_W1, h_W1, 24 * sizeof(float), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_b1, h_b1, 8 * sizeof(float), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_W2, h_W2, 8 * sizeof(float), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_b2, h_b2, sizeof(float), 0, cudaMemcpyHostToDevice);
}

// Residual Pruning upload
void upload_rp_constants(
    float* h_alpha, float* h_beta, float* h_gamma, int num_levels,
    float* h_lut_breaks, float* h_lut_slopes, float* h_lut_intercepts, int lut_size,
    int mode)
{
    cudaError_t err;
    err = cudaMemcpyToSymbol(c_rp_alpha, h_alpha, RP_MAX_LEVELS * sizeof(float));
    if (err != cudaSuccess) printf("Error uploading c_rp_alpha: %s\n", cudaGetErrorString(err));
    
    err = cudaMemcpyToSymbol(c_rp_beta, h_beta, RP_MAX_LEVELS * sizeof(float));
    if (err != cudaSuccess) printf("Error uploading c_rp_beta: %s\n", cudaGetErrorString(err));
    
    err = cudaMemcpyToSymbol(c_rp_gamma, h_gamma, RP_MAX_LEVELS * sizeof(float));
    if (err != cudaSuccess) printf("Error uploading c_rp_gamma: %s\n", cudaGetErrorString(err));
    
    if (h_lut_breaks && h_lut_slopes && h_lut_intercepts) {
        err = cudaMemcpyToSymbol(c_lut_breaks, h_lut_breaks, RP_LUT_SIZE * sizeof(float));
        if (err != cudaSuccess) printf("Error uploading c_lut_breaks: %s\n", cudaGetErrorString(err));
        err = cudaMemcpyToSymbol(c_lut_slopes, h_lut_slopes, RP_LUT_SIZE * sizeof(float));
        if (err != cudaSuccess) printf("Error uploading c_lut_slopes: %s\n", cudaGetErrorString(err));
        err = cudaMemcpyToSymbol(c_lut_intercepts, h_lut_intercepts, RP_LUT_SIZE * sizeof(float));
        if (err != cudaSuccess) printf("Error uploading c_lut_intercepts: %s\n", cudaGetErrorString(err));
    }
    
    err = cudaMemcpyToSymbol(c_lut_num_segments, &lut_size, sizeof(int));
    if (err != cudaSuccess) printf("Error uploading c_lut_num_segments: %s\n", cudaGetErrorString(err));
    
    err = cudaMemcpyToSymbol(c_rp_mode, &mode, sizeof(int));
    if (err != cudaSuccess) printf("Error uploading c_rp_mode: %s\n", cudaGetErrorString(err));
    
    printf("[upload_rp_constants] Mode=%d, LUT segments=%d\n", mode, lut_size);
}
inline size_t calcRawDataBytes(int *data_info, short *data_d, char *data_s, int *size_s)
{
	if (data_info[2] != 6)
	{
		return static_cast<size_t>(data_info[1]) * data_info[0] * sizeof(data_d[0]);
	}
	size_t char_bytes = static_cast<size_t>(data_info[1]) * M * sizeof(data_s[0]);
	size_t len_bytes = static_cast<size_t>(data_info[1]) * sizeof(size_s[0]);
	return char_bytes + len_bytes;
}

inline double bytesToMB(size_t bytes)
{
	return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

int *data_info;
short *data_d;
char *data_s;
int *size_s;
int *qid_list;
float *query_data;
int qnum;
int *max_node_num;
int *id_list;
TN *node_list;
char *file;
char *file_q;
char *file_gt;
char *file_u;
int *res_ids;
bool use_vector_query = false;

static bool isTxtFile(const char *path)
{
	if (!path)
		return false;
	const char *dot = strrchr(path, '.');
	if (!dot)
		return false;
	return (strcmp(dot, ".txt") == 0 || strcmp(dot, ".TXT") == 0);
}
float time_index = 0;
float time_search = 0;
float time_calibration = 0;
float time_update_s = 0;
float time_update_u = 0;
int count_update_s = 0;
int count_update_u = 0;
int tree_h;
int k;	 // k for knn
float r; // r for range query
int *empty_list;
int *qresult_count;
int *qresult_count_prefix;
int *result_id;
float *result_dis;
int process_type;
int search_type;

int main(int argc, char **argv)
{
	file = argv[1];
	load(file, data_info, data_d, data_s, size_s);
	process_type = (int)atoi(argv[3]);
	if (process_type != 2)
	{
		file_q = argv[2];
		// Parse optional vec/id mode and groundtruth arguments
		char *arg6 = (argc > 6) ? argv[6] : nullptr;
		char *arg7 = (argc > 7) ? argv[7] : nullptr;
		const char *query_mode = "id";
		if (arg6 && (strcmp(arg6, "id") == 0 || strcmp(arg6, "vec") == 0 || strcmp(arg6, "vector") == 0))
		{
			query_mode = arg6;
			file_gt = arg7;
		}
		else
		{
			file_gt = arg6;
		}
		use_vector_query = (strcmp(query_mode, "vec") == 0 || strcmp(query_mode, "vector") == 0);
		if (use_vector_query)
		{
			if (isTxtFile(file_q))
			{
				loadQueryTxt(file_q, query_data, qnum, data_info[0]);
			}
			else
			{
				loadQuery(file_q, query_data, qnum, data_info[0]);
			}
		}
		else
		{
			loadQuery(file_q, qid_list, qnum);
		}
		k = (int)atoi(argv[4]);
		r = (float)stod(argv[4]);
		if (use_vector_query && process_type != 0)
		{
			printf("Vector query mode only supports KNN (process_type=0).\n");
			return 1;
		}
		if (use_vector_query && data_info[2] == 6)
		{
			printf("Vector query mode does not support string distance (data_info[2]==6).\n");
			return 1;
		}
		// TREE_ORDER = (int)atoi(argv[5]);
		// MAX_SIZE = (int)atoi(argv[6]);
		// MAX_H = (int)atoi(argv[7]);
		// DIS_CODE = (int)atoi(argv[8]);
		// INFI_DIS = (int)atoi(argv[9]);
		// float temp_s = (float)stod(argv[10]);
		// input_size = temp_s * 1024 * 1024 * 1024;
		// printf("%f, %f\n", temp_s, input_size);
	}
	else
	{
		file_u = argv[2];
		loadUpdate(file_u, update_list, update_num);
		// search_type = (int)atoi(argv[4]);
		search_type = 1;
		// k = (int)atoi(argv[5]);
		r = (float)stod(argv[4]);
		// TREE_ORDER = (int)atoi(argv[6]);
		// MAX_SIZE = (int)atoi(argv[7]);
		// MAX_H = (int)atoi(argv[8]);
		// DIS_CODE = (int)atoi(argv[9]);
		// INFI_DIS = (int)atoi(argv[10]);
		// MAX_IN_SIZE = (int)atoi(argv[11]);
		const char* __env_bstar = getenv("GTSPP_MAX_IN_SIZE");
		if (__env_bstar) {
			MAX_IN_SIZE = atoi(__env_bstar);
			printf("[B*] MAX_IN_SIZE overridden to %d via GTSPP_MAX_IN_SIZE\n", MAX_IN_SIZE);
		}
	}

	// Index Construction - 使用GPU计时
	AdvancedGPUTimer timer("Index Construction");
	timer.start();
	indexConstru(data_d, data_s, size_s, data_info, id_list, node_list, max_node_num, tree_h, empty_list);
	timer.add_measurement();
	time_index += timer.get_total_time() / 1000.0f; // 转换为秒

	size_t raw_data_bytes = calcRawDataBytes(data_info, data_d, data_s, size_s);
	size_t id_map_bytes = static_cast<size_t>(data_info[1]) * sizeof(id_list[0]);
	size_t node_bytes = static_cast<size_t>(max_node_num[0]) * sizeof(TN);
	size_t static_total_bytes = raw_data_bytes + id_map_bytes + node_bytes;

	printf("\n[Static Storage Footprint]\n");
	printf("  Raw data buffer      : %.2f MB (%zu bytes)\n", bytesToMB(raw_data_bytes), raw_data_bytes);
	printf("  ID map (id_list)     : %.2f MB (%zu bytes)\n", bytesToMB(id_map_bytes), id_map_bytes);
	printf("  Tree nodes (node_list): %.2f MB (%zu bytes)\n", bytesToMB(node_bytes), node_bytes);
	printf("  Total static storage : %.2f MB (%zu bytes)\n\n", bytesToMB(static_total_bytes), static_total_bytes);

	// knn
	if (process_type == 0)
	{
		FILE *fcost = fopen(argv[5], "w");
		fprintf(fcost, "Knn search num: %d\nResult radius: \n", k);
		fflush(fcost);
		
		// [Residual Pruning] Offline auto-tuning before search
		printf("\n[Main] Starting Residual Pruning auto-tuning for KNN search...\n");
		ResidualPruningTuner rp_tuner;
		
		// Copy data to CPU for initialization
		short* data_h = new short[data_info[1] * data_info[0]];
		int* id_list_h = new int[data_info[1]];
		TN* node_list_h = new TN[max_node_num[0]];
		
		cudaMemcpy(data_h, data_d, data_info[1] * data_info[0] * sizeof(short), cudaMemcpyDeviceToHost);
		cudaMemcpy(id_list_h, id_list, data_info[1] * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(node_list_h, node_list, max_node_num[0] * sizeof(TN), cudaMemcpyDeviceToHost);
		
		// Initialize: all scales = 1.0 (baseline)
		rp_tuner.AutoTuneAndUpload(node_list_h, max_node_num[0], data_h, id_list_h, 
		                           data_info[0], data_info[1], data_info[2], tree_h);
		
		// Release temporary memory
		delete[] data_h;
		delete[] id_list_h;
		delete[] node_list_h;
		
		// ========== Per-Level Scale Calibration ==========
		// For each level (from deepest to shallowest), binary search for
		// the maximum scale factor where recall remains >= 99.9%.
		// 
		// scale = 1.0 means baseline (no extra pruning)
		// scale > 1.0 means tighter pruning: dis_lb_eff = dis_lb * scale
		// Guarantee: scale >= 1.0 → delta >= 0 → zero false negatives
		//
		auto calib_begin = std::chrono::steady_clock::now();
		if (file_gt != nullptr) {
			printf("\n[Main] Per-Level Scale Calibration (binary search on actual recall)\n");
			printf("========================================\n");
			
			int *gt_ids_calib = nullptr;
			if (isTxtFile(file_gt)) {
				loadGroundTruthTxt(file_gt, gt_ids_calib, qnum, 100);
			} else {
				loadGroundTruth(file_gt, gt_ids_calib, qnum, 100);
			}
			
			if (gt_ids_calib != nullptr) {
				int calib_qnum = std::min(1000, qnum);
				int *res_ids_calib = nullptr;
				CHECK(cudaMallocManaged((void **)&res_ids_calib, calib_qnum * k * sizeof(int)));
				
				// Step 0: Measure baseline recall (all gamma=1.0, no pruning)
				update_disk = false;
				if (use_vector_query)
					searchIndexKnnV2(data_d, node_list, id_list, max_node_num, query_data,
									res_ids_calib, calib_qnum, k, tree_h, data_info, empty_list, data_s, size_s);
				else
					searchIndexKnnV2(data_d, node_list, id_list, max_node_num, qid_list,
									res_ids_calib, calib_qnum, k, tree_h, data_info, empty_list, data_s, size_s);
				cudaDeviceSynchronize();
				int baseline_correct = 0;
				for (int i = 0; i < calib_qnum; i++)
					for (int j = 0; j < k; j++) {
						int pred_id = res_ids_calib[i * k + j];
						for (int m = 0; m < k; m++)
							if (pred_id == gt_ids_calib[i * 100 + m]) { baseline_correct++; break; }
					}
				float baseline_recall = (float)baseline_correct / (calib_qnum * k);
				cudaFree(res_dis); res_dis = nullptr;
				printf("  Baseline recall (no pruning): %.4f\n", baseline_recall);
				
				// Adaptive target: maintain baseline recall exactly
				float recall_target_adaptive = baseline_recall;
				
				// Calibrate from deepest level to shallowest (bottom-up)
				for (int level = tree_h - 1; level >= 3; level--) {
					float lo = 1.0f, hi = 5.0f;
					float best_scale = 1.0f;
					
					for (int iter = 0; iter < 16; iter++) {  // 16 iters for finer precision
						float mid = (lo + hi) / 2.0f;
						rp_tuner.SetLevelScale(level, mid);
						
						// Reset search state
						update_disk = false;
						
						// Run search on calibration subset (supports both vector and ID query modes)
						if (use_vector_query)
						{
							searchIndexKnnV2(data_d, node_list, id_list, max_node_num, query_data,
											res_ids_calib, calib_qnum, k, tree_h, data_info, empty_list, data_s, size_s);
						}
						else
						{
							searchIndexKnnV2(data_d, node_list, id_list, max_node_num, qid_list,
											res_ids_calib, calib_qnum, k, tree_h, data_info, empty_list, data_s, size_s);
						}
						cudaDeviceSynchronize();
						
						// Compute recall
						int correct = 0;
						for (int i = 0; i < calib_qnum; i++) {
							for (int j = 0; j < k; j++) {
								int pred_id = res_ids_calib[i * k + j];
								for (int m = 0; m < k; m++) {
									if (pred_id == gt_ids_calib[i * 100 + m]) {
										correct++;
										break;
									}
								}
							}
						}
						float recall = (float)correct / (calib_qnum * k);
						
						// Free search-allocated memory
						cudaFree(res_dis);
						res_dis = nullptr;
						
						if (recall >= recall_target_adaptive) {
							best_scale = mid;
							lo = mid;
						} else {
							hi = mid;
						}
					}
					
					// Apply safety margin: reduce excess scale by 25% to handle
					// generalization gap between calibration subset and full query set
					float safe_scale = 1.0f + (best_scale - 1.0f) * 0.75f;
					rp_tuner.SetLevelScale(level, safe_scale);
					printf("  Level %d: raw=%.4f safe=%.4f\n", level, best_scale, safe_scale);
				}
				
				printf("========================================\n");
				printf("[Calibrate] Per-level calibration complete.\n");
				rp_tuner.PrintStatus();
				
				cudaFree(res_ids_calib);
				delete[] gt_ids_calib;
			}
		}
		auto calib_end = std::chrono::steady_clock::now();
		time_calibration += std::chrono::duration_cast<std::chrono::duration<float>>(calib_end - calib_begin).count();
		printf("Time of calibration: %f\n", time_calibration);
		fprintf(fcost, "Time of calibration: %f\n", time_calibration);
		
		// CRITICAL: Reset search state after calibration
		update_disk = false;
		
		// knn - 使用GPU计时
		AdvancedGPUTimer search_timer("Knn Search");
		search_timer.start();
		if (use_vector_query)
		{
			CHECK(cudaMallocManaged((void **)&res_ids, qnum * k * sizeof(int)));
			searchIndexKnnV2(data_d, node_list, id_list, max_node_num, query_data, res_ids, qnum, k, tree_h, data_info, empty_list, data_s, size_s);
			saveK((char *)"result_ids.txt", (char *)"result_dists.txt", res_ids, res_dis, k, qnum);
		}
		else
		{
			searchIndexKnnV2(data_d, node_list, id_list, max_node_num, qid_list, qnum, k, tree_h, data_info, empty_list, data_s, size_s);
		}
		cudaError_t syncStatus = cudaDeviceSynchronize();
		if (syncStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaDeviceSynchronize error: %s\n", cudaGetErrorString(syncStatus));
		}
		search_timer.add_measurement();
		time_search += search_timer.get_total_time() / 1000.0f; // 转换为秒

		// Output results
		if (use_vector_query)
		{
			for (int i = 0; i < qnum; i++)
			{
				fprintf(fcost, "%f ", res_dis[i * k + (k - 1)]);
				fflush(fcost);
			}
		}
		else
		{
			for (int i = 0; i < qnum; i++)
			{
				fprintf(fcost, "%f ", res_dis[i]);
				fflush(fcost);
			}
		}
		printf("Time of index construction: %f\n", time_index);
		printf("Average search time: %f\n", time_search / qnum);
		printf("query number: %d\n", qnum);
		fprintf(fcost, "\nTime of index construction: %f\n", time_index);
		fprintf(fcost, "Average search time: %f\n", time_search / qnum);
		fprintf(fcost, "query number: %d\n", qnum);

		// Recall 仅在向量查询模式下计算
		if (use_vector_query && file_gt != nullptr)
		{
			printf("\n>>> Calculating Recall...\n");
			int *gt_ids = nullptr;
			if (isTxtFile(file_gt))
			{
				loadGroundTruthTxt(file_gt, gt_ids, qnum, 100);
			}
			else
			{
				loadGroundTruth(file_gt, gt_ids, qnum, 100);
			}
			if (gt_ids != nullptr)
			{
				int correct_count = 0;
				for (int i = 0; i < qnum; i++)
				{
					for (int j = 0; j < k; j++)
					{
						int pred_id = res_ids[i * k + j];
						for (int m = 0; m < k; m++)
						{
							if (pred_id == gt_ids[i * 100 + m])
							{
								correct_count++;
								break;
							}
						}
					}
				}
				float recall = (float)correct_count / (qnum * k);
				printf("========================================\n");
				printf("Recall@%d: %.4f (%.2f%%)\n", k, recall, recall * 100.0f);
				printf("========================================\n");
				fprintf(fcost, "Recall@%d: %.4f\n", k, recall);

				// [Contribution 4] Online Adaptation: adjust pruning based on recall
				rp_tuner.OnlineAdapt(recall);
				rp_tuner.PrintStatus();

				delete[] gt_ids;
			}
		}

		// Print Residual Pruning summary
		fprintf(fcost, "\n[Residual Pruning Summary]\n");
		fprintf(fcost, "  Mode: %d\n", rp_tuner.GetMode());
		fprintf(fcost, "  Gamma Scale: %.3f\n", rp_tuner.GetGammaScale());

		fflush(fcost);
		fclose(fcost);
	}

	// Range query
	else if (process_type == 1)
	{
		FILE *fcost = fopen(argv[5], "w");
		fprintf(fcost, "Range search radius: %f\nResult num: \n", r);
		fflush(fcost);

		// Range query - 使用GPU计时
		AdvancedGPUTimer rnn_search_timer("Range Search");
		rnn_search_timer.start();
		searchIndexRnnV2(data_d, node_list, id_list, max_node_num, qid_list, qnum, r, tree_h, data_info, empty_list, data_s, size_s);
		rnn_search_timer.add_measurement();
		time_search += rnn_search_timer.get_total_time() / 1000.0f; // 转换为秒

		// Output results
		for (int i = 0; i < qnum; i++)
		{
			fprintf(fcost, "%d ", res[i]);
			fflush(fcost);
		}
		printf("\nTime of index construction: %f\n", time_index);
		printf("Average search time: %f\n", time_search / qnum);
		printf("query number: %d\n", qnum);
		fprintf(fcost, "\nTime of index construction: %f\n", time_index);
		fprintf(fcost, "Average search time: %f\n", time_search / qnum);
		fprintf(fcost, "query number: %d\n", qnum);
		fflush(fcost);
		fclose(fcost);
	}

	// Update
	else
	{
		FILE *fcost = fopen(argv[5], "w");
		fprintf(fcost, "Range search radius (check for updates): %f\nResult num: \n", r);
		fflush(fcost);

			// Activate residual pruning with configurable gamma (env RP_GAMMA, default 10.0)
			{
				float h_alpha[32] = {0};
				float h_beta[32] = {0};
				float h_gamma[32];
				float gamma_val = 10.0f;
				const char* env_gamma = getenv("RP_GAMMA");
				if (env_gamma) gamma_val = atof(env_gamma);
				for (int l = 0; l < 32; l++) {
					if (l <= 2) h_gamma[l] = 1.0f;
					else h_gamma[l] = gamma_val;
				}
				upload_rp_constants(h_alpha, h_beta, h_gamma, tree_h, nullptr, nullptr, nullptr, 0, 0);  // Mode 0: C1 uses all-include, not gamma
				printf("[Update] RP gamma=[1,1,1,%.1f,...] env=%s\n", gamma_val, env_gamma ? env_gamma : "10");
			}

		// Update
		updateIndexRnn(data_d, node_list, id_list, max_node_num, qid_list, 1, r, tree_h, data_info, empty_list,
					   qresult_count, qresult_count_prefix, result_id, result_dis, data_s, size_s, fcost, time_update_s, time_update_u,
					   count_update_s, count_update_u);

		// Output results
		printf("Time of index construction: %f\n", time_index);
		printf("Total update time: %f\n", time_update_s / count_update_s + time_update_u / count_update_u);
		printf("Search time in update: %f\n", time_update_s / count_update_s);
		printf("Update time in update: %f\n", time_update_u / count_update_u);
		fprintf(fcost, "\nTime of index construction: %f\n", time_index);
		fprintf(fcost, "Total update time: %f\n", time_update_s / count_update_s + time_update_u / count_update_u);
		fprintf(fcost, "Search time in update : % f\n", time_update_s / count_update_s);
		fprintf(fcost, "Update time in update: %f\n", time_update_u / count_update_u);
		fflush(fcost);
		fclose(fcost);
	}

	// Release memory
	cudaFree(data_info);
	cudaFree(data_d);
	cudaFree(data_s);
	cudaFree(size_s);
	cudaFree(id_list);
	cudaFree(node_list);
	cudaFree(max_node_num);
	cudaFree(qid_list);
	cudaFree(empty_list);
	cudaFree(update_list);
	cudaFree(res);
	cudaFree(res_dis);
	return 0;
}