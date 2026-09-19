#pragma once
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <stdio.h>
#include "config.cuh"
using namespace std;
#define M 109

// Split the line
void split(const string str, vector<string> &res, const char pattern)
{
    istringstream is(str);
    string temp;
    while (getline(is, temp, pattern))
        res.push_back(temp);
    return;
}

// Load data file (保持不变)
void load(char *file, int *&data_info, short *&data_d, char *&data_s, int *&size_s)
{
    ifstream in(file);
    if (!in.is_open())
    {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    cout << "Loading data file..." << endl;
    string line;
    int i = 0;
    int j = 0;
    vector<string> res;
    cudaMallocManaged((void **)&data_info, 3 * sizeof(int));
    while (getline(in, line))
    {
        if (i == 0)
        { 
            split(line, res, ' ');
            for (auto r : res)
            {
                stringstream ss(r);
                int number;
                ss >> number;
                if (j == 0) data_info[j] = number;
                if (j == 1) data_info[j] = number;
                if (j == 2) data_info[j] = number;
                j++;
            }
            if (data_info[2] != 6)
                cudaMallocManaged((void **)&data_d, data_info[1] * data_info[0] * sizeof(short));
            else
            {
                cudaMallocManaged((void **)&data_s, data_info[1] * M * sizeof(char));
                cudaMallocManaged((void **)&size_s, data_info[1] * sizeof(int));
            }
        }
        else
        {
            if (data_info[2] != 6)
            {
                split(line, res, ' ');
                for (auto r : res)
                {
                    stringstream ss(r);
                    float number;
                    ss >> number;
                    *(data_d + (i - 1) * data_info[0] + j) = number;
                    j++;
                }
            }
            else
            {
                const char *temp = line.data();
                memcpy(data_s + (i - 1) * M, temp, strlen(temp));
                size_s[i - 1] = strlen(temp);
            }
        }
        res.clear();
        j = 0;
        i++;
    }
    in.close();
}

// ==========================================
// 核心修改区：通过函数重载实现“ID查询”与“向量查询”并行
// ==========================================

// [兼容旧模式] 1. 加载文本格式的 Query ID List
// 只要传入的是 int*，编译器就会自动选这个函数
void loadQuery(char *file, int *&qid, int &qnum)
{
    ifstream in(file);
    if (!in.is_open())
    {
        std::cout << "open file error: " << file << std::endl;
        exit(-1);
    }

    cout << "Loading Query IDs (Legacy Text Mode)..." << endl;

    string line;
    int i = 0;
    while (getline(in, line))
    {
        if (i == 0)
        { 
            stringstream ss(line);
            int number;
            ss >> number;
            cudaMallocManaged((void **)&qid, number * sizeof(int));
            qnum = number;
        }
        else
        { 
            stringstream ss(line);
            int number;
            ss >> number;
            qid[i - 1] = number;
        }
        i++;
    }
    in.close();
}

// [H100标准模式] 2. 加载二进制格式的 Query Vectors (Float)
// 只要传入的是 float*，编译器就会自动选这个函数
void loadQuery(char *file, float *&query_data, int &qnum, int dim)
{
    FILE *f = fopen(file, "rb");
    if (!f) {
        std::cout << "Error opening binary query file: " << file << std::endl;
        exit(-1);
    }

    // 自动计算 query 数量
    fseek(f, 0, SEEK_END);
    long fileSize = ftell(f);
    fseek(f, 0, SEEK_SET);

    // 假设是纯二进制 float (无头文件)
    qnum = fileSize / (dim * sizeof(float));

    std::cout << "Loading Query Vectors (Standard Binary Mode): " << qnum << " queries..." << std::endl;

    // 使用统一内存，CPU/GPU 均可访问
    cudaMallocManaged((void **)&query_data, qnum * dim * sizeof(float));

    if (fread(query_data, sizeof(float), qnum * dim, f) != qnum * dim) {
        std::cout << "Error reading query data (size mismatch)." << std::endl;
    }

    fclose(f);
}

// [新增] 2.1 加载 TXT 格式的 Query Vectors (Float)
// 支持两种格式：
//   A) 第一行是 qnum，后续共有 qnum*dim 个 float
//   B) 直接提供 qnum*dim 个 float（不含表头）
void loadQueryTxt(char *file, float *&query_data, int &qnum, int dim)
{
    ifstream in(file);
    if (!in.is_open())
    {
        std::cout << "Error opening query txt file: " << file << std::endl;
        exit(-1);
    }

    std::vector<float> vals;
    vals.reserve(1024);
    std::string line;
    int header_qnum = -1;
    int qnum_count = 0;
    bool header_checked = false;

    while (std::getline(in, line))
    {
        if (line.empty())
            continue;

        std::vector<float> tokens;
        tokens.reserve(dim + 1);
        std::istringstream iss(line);
        float v;
        while (iss >> v)
        {
            tokens.push_back(v);
        }
        if (tokens.empty())
            continue;

        if (!header_checked)
        {
            header_checked = true;
            if (tokens.size() == 1)
            {
                float t = tokens[0];
                int ti = static_cast<int>(t);
                if (t == static_cast<float>(ti) && ti > 0)
                {
                    header_qnum = ti;
                    continue;
                }
            }
        }

        if (static_cast<int>(tokens.size()) == dim + 1)
        {
            for (int i = 1; i <= dim; ++i)
                vals.push_back(tokens[i]);
            qnum_count++;
        }
        else if (static_cast<int>(tokens.size()) == dim)
        {
            for (int i = 0; i < dim; ++i)
                vals.push_back(tokens[i]);
            qnum_count++;
        }
        else
        {
            std::cout << "Error: query txt line dimension mismatch. Expected " << dim
                      << " or " << (dim + 1) << " values, got " << tokens.size() << std::endl;
            exit(-1);
        }
    }
    in.close();

    if (vals.empty())
    {
        std::cout << "Error: empty query txt file." << std::endl;
        exit(-1);
    }

    if (header_qnum > 0 && header_qnum != qnum_count)
    {
        std::cout << "Warning: query txt header mismatch. header=" << header_qnum
                  << ", parsed=" << qnum_count << std::endl;
    }
    qnum = qnum_count;

    std::cout << "Loading Query Vectors (TXT Mode): " << qnum << " queries..." << std::endl;
    cudaMallocManaged((void **)&query_data, qnum * dim * sizeof(float));
    size_t total = static_cast<size_t>(qnum) * dim;
    for (size_t i = 0; i < total; ++i)
    {
        query_data[i] = vals[i];
    }
}

// [新增] 3. 加载二进制格式的 Ground Truth IDs
void loadGroundTruth(char *file, int *&gt_ids, int qnum, int k)
{
    FILE *f = fopen(file, "rb");
    if (!f) {
        std::cout << "Error opening ground truth file: " << file << std::endl;
        exit(-1);
    }

    // 分配 CPU 内存用于验证
    gt_ids = new int[qnum * k];

    std::cout << "Loading Ground Truth (" << qnum << " x " << k << ")..." << std::endl;

    // 读取纯二进制 int32
    if (fread(gt_ids, sizeof(int), qnum * k, f) != qnum * k) {
        std::cout << "Warning: GT file size mismatch." << std::endl;
    }

    fclose(f);
}

// [新增] 4. 加载 TXT 格式的 Ground Truth IDs
// 支持两种格式：
//   A) 第一行是 qnum，后续共有 qnum*k 个 int
//   B) 直接提供 qnum*k 个 int（不含表头）
void loadGroundTruthTxt(char *file, int *&gt_ids, int qnum, int k)
{
    ifstream in(file);
    if (!in.is_open())
    {
        std::cout << "Error opening ground truth txt file: " << file << std::endl;
        exit(-1);
    }

    std::vector<int> vals;
    vals.reserve(qnum * k + 1);
    int v;
    while (in >> v)
    {
        vals.push_back(v);
    }
    in.close();

    if (vals.empty())
    {
        std::cout << "Error: empty ground truth txt file." << std::endl;
        exit(-1);
    }

    size_t expected = static_cast<size_t>(qnum) * k;
    size_t offset = 0;
    if (vals.size() == expected + 1 && vals[0] == qnum)
    {
        offset = 1;
    }
    else if (vals.size() != expected)
    {
        std::cout << "Warning: GT txt count mismatch. Got " << vals.size()
                  << ", expected " << expected << " (or " << expected + 1 << " with header)." << std::endl;
    }

    gt_ids = new int[qnum * k];
    size_t copy_n = std::min(expected, vals.size() - offset);
    for (size_t i = 0; i < copy_n; ++i)
    {
        gt_ids[i] = vals[offset + i];
    }
    // 若不足，剩余填充为 -1
    for (size_t i = copy_n; i < expected; ++i)
    {
        gt_ids[i] = -1;
    }
}

// Save result file for knn (已优化 IO)
void saveK(char *filenamer, char *filenamed, int *res_id, float *dis, int k, int qnum)
{
    FILE *resr = fopen(filenamer, "w");
    FILE *resd = fopen(filenamed, "w");

    for (unsigned i = 0; i < qnum; i++)
    {
        for (unsigned j = 0; j < k; j++)
        {
            fprintf(resr, "%d ", res_id[i * k + j]);
            fprintf(resd, "%f ", dis[i * k + j]);
        }
        fprintf(resr, "\n");
        fprintf(resd, "\n");
    }
    fclose(resr);
    fclose(resd);
}

// Save result file for rnn
void saveR(char *filenamer, char *filenamed, int *res_id, float *dis, int *qresult_count, int *qresult_count_prefix, int qnum)
{
    FILE *resr = fopen(filenamer, "a");
    FILE *resd = fopen(filenamed, "a");

    for (unsigned i = 0; i < qnum; i++)
    {
        for (unsigned j = qresult_count_prefix[i]; j < qresult_count_prefix[i] + qresult_count[i]; j++)
        {
            fprintf(resr, "%d ", res_id[j]);
            fprintf(resd, "%f ", dis[j]);
        }
        fprintf(resr, "\n");
        fprintf(resd, "\n");
    }
    fclose(resr);
    fclose(resd);
}