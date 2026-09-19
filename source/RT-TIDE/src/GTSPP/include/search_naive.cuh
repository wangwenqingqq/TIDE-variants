// Naive search (search array directly)
// Created on 24-01-05

#pragma once
#include <iostream>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <math_functions.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include "file.cuh"
#include "tree.cuh"
#include "config.cuh"

// Result struct
typedef struct Object
{
	float *dis_q;  // distance for each query
	int *res_id_q; // result id for each query
} Obj;

// Compute the distance (num type)
__global__ void searchD(int *data_info, short *data_d, int qid, Obj obj, int in_size, int *insert_list)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int idx = id; idx < in_size; idx += total_num)
	{
		float result = 0;

		if (data_info[2] == 2)
		{ // L2 distance
			for (int i = 0; i < data_info[0]; i++)
			{
				{ float _d = data_d[insert_list[idx] * data_info[0] + i] - data_d[qid * data_info[0] + i]; result += _d * _d; }
			}

			result = sqrtf(result);
		}
		else if (data_info[2] == 1)
		{ // L1 distance
			for (int i = 0; i < data_info[0]; i++)
			{
				result += abs(data_d[insert_list[idx] * data_info[0] + i] - data_d[qid * data_info[0] + i]);
			}
		}
		else if (data_info[2] == 0)
		{ // Max value
			float temp = 0;

			for (int i = 0; i < data_info[0]; i++)
			{
				temp = abs(data_d[insert_list[idx] * data_info[0] + i] - data_d[qid * data_info[0] + i]);
				if (temp > result)
					result = temp;
			}
		}
		else if (data_info[2] == 5)
		{
			float sa1 = 0, sa2 = 0, sa3 = 0;
			for (int i = 0; i < data_info[0]; i++)
			{
				sa1 += data_d[insert_list[idx] * data_info[0] + i] * data_d[insert_list[idx] * data_info[0] + i];
				sa2 += data_d[qid * data_info[0] + i] * data_d[qid * data_info[0] + i];
				sa3 += data_d[insert_list[idx] * data_info[0] + i] * data_d[qid * data_info[0] + i];
			}
			sa1 = sqrtf(sa1);
			sa2 = sqrtf(sa2);
			if (sa1 * sa2 == 0)
			{
				printf("Error!!!\n");
			}
			result = sa3 / (sa1 * sa2);
			if (result > 1)
			{
				result = 0.99999999999999999;
			}
			result = abs(acos(result) * 180 / 3.1415926);
		}

		obj.dis_q[idx] = result;
		obj.res_id_q[idx] = idx;
	}
}

// Compute the distance (text type)
__global__ void searchS(int *data_info, char *data_s, int *size_s, int qid, Obj obj, int in_size, int *insert_list)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int idx = id; idx < in_size; idx += total_num)
	{ // Edit distance
		float result = 0;
		int n = size_s[insert_list[idx]];
		int m = size_s[qid];

		int table[M][M];

		if (n == 0)
			result = m;
		if (m == 0)
			result = n;
		if (n != 0 && m != 0)
		{
			for (int i = 0; i <= n; i++)
				table[i][0] = i;
			for (int j = 0; j <= m; j++)
				table[0][j] = j;
			for (int i = 1; i <= n; i++)
			{
				for (int j = 1; j <= m; j++)
				{
					int cost = (data_s[insert_list[idx] * M + i - 1] == data_s[qid * M + j - 1]) ? 0 : 1;
					table[i][j] = 1 + min(table[i - 1][j], table[i][j - 1]);
					table[i][j] = min(table[i - 1][j - 1] + cost, table[i][j]);
				}
			}
			result = table[n][m];
		}

		obj.dis_q[idx] = result;
		obj.res_id_q[idx] = idx;
	}
}

// Check if the range is satisfied
__global__ void check(int *accu_upper, float *dis_q, float r, int num)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int idx = id; idx < num; idx += total_num)
	{
		if (dis_q[idx] <= r)
		{
			accu_upper[idx] = 1;
		}
		else
		{
			accu_upper[idx] = 0;
		}
	}
}

// Compute accu array for rnn
__global__ void getAccu(int *accu, int *accu_upper, int level_curr, int num)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int idx = id; idx < num; idx += total_num)
	{
		int i = idx - (1 << (level_curr - 1));

		if (i >= 0)
		{
			accu[idx] = accu_upper[idx] + accu_upper[i];
		}
		else
		{
			accu[idx] = accu_upper[idx];
		}
	}
}

// Update upper array of accu
__global__ void updateUpp(int *accu, int *accu_upper, int num)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int idx = id; idx < num; idx += total_num)
	{
		accu_upper[idx] = accu[idx];
	}
}

// Get rnn result according to accu
__global__ void getRnn(int *accu, Obj obj, Obj obj_r, int num)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int idx = id; idx < num; idx += total_num)
	{
		if (accu[idx] > 0 && idx == 0)
		{
			int i = accu[idx];
			obj_r.res_id_q[i - 1] = obj.res_id_q[idx];
			obj_r.dis_q[i - 1] = obj.dis_q[idx];
		}
		else if (idx > 0)
		{
			if (accu[idx] > accu[idx - 1])
			{
				int i = accu[idx];
				obj_r.res_id_q[i - 1] = obj.res_id_q[idx];
				obj_r.dis_q[i - 1] = obj.dis_q[idx];
			}
		}
	}
}

// range query
// Pre-allocated workspace for searchNaiveRnn
static float *naive_ws_dis = nullptr;
static int *naive_ws_id = nullptr;
static int *naive_ws_accu = nullptr;
static int naive_ws_cap = 0;

void searchNaiveRnn(int *data_info, Obj &obj_r, short *data_d, char *data_s, int *size_s, int qid, int in_size, float r,
					int *insert_list, int *rnum)
{
	Obj obj;
	int block_num = (in_size + THREAD_NUM - 1) / THREAD_NUM;

	// Workspace reuse: pre-allocate once, grow as needed
	if (naive_ws_cap < in_size) {
		if (naive_ws_dis) cudaFree(naive_ws_dis);
		if (naive_ws_id) cudaFree(naive_ws_id);
		if (naive_ws_accu) cudaFree(naive_ws_accu);
		cudaMalloc((void **)&naive_ws_dis, in_size * 2 * sizeof(float));
		cudaMalloc((void **)&naive_ws_id, in_size * 2 * sizeof(int));
		cudaMalloc((void **)&naive_ws_accu, in_size * 2 * sizeof(int));
		naive_ws_cap = in_size * 2;
	}
	obj.dis_q = naive_ws_dis;
	obj.res_id_q = naive_ws_id;
	int *accu = naive_ws_accu;

	if (data_info[2] != 6)
	{ // num type
		searchD<<<block_num, THREAD_NUM>>>(data_info, data_d, qid, obj, in_size, insert_list);
	}
	else
	{ // text type
		searchS<<<block_num, THREAD_NUM>>>(data_info, data_s, size_s, qid, obj, in_size, insert_list);
	}
	cudaDeviceSynchronize();
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "Search naive error: %s\n", cudaGetErrorString(cudaStatus));

	check<<<block_num, THREAD_NUM>>>(accu, obj.dis_q, r, in_size);
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "check error: %s\n", cudaGetErrorString(cudaStatus));

	// Manual inclusive scan (avoids thrust abort on stale CUDA errors)
	{
		int *h_accu = (int*)malloc(in_size * sizeof(int));
		cudaMemcpy(h_accu, accu, in_size * sizeof(int), cudaMemcpyDeviceToHost);
		for (int si = 1; si < in_size; si++) h_accu[si] += h_accu[si-1];
		rnum[0] = h_accu[in_size - 1];
		cudaMemcpy(accu, h_accu, in_size * sizeof(int), cudaMemcpyHostToDevice);
		free(h_accu);
	}

	getRnn<<<block_num, THREAD_NUM>>>(accu, obj, obj_r, in_size);
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "getRnn error: %s\n", cudaGetErrorString(cudaStatus));

	// Workspace reused - no free
}