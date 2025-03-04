#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include<math.h>
#include<time.h>
using namespace std;

int ReverseInt(int i);
void read_Mnist_Label(string filename, vector<double>& labels);
void read_Mnist_Images(string filename, vector<vector<double>>& images);
double sigmoid(double z);
double sigmoid_prime(double z);//激活函数的导函数
vector<double> sigmoid_prime_vec(vector<double> z);
void dump_double_list(vector<double> list);
void vectorMulti(vector<double>& list, double t);
vector<double> torchMul(vector<double> t1, vector<double> t2);//对应位置相乘
vector<vector<double>> transposeInPlace(vector<vector<double>> m);//返回v的转置矩阵
