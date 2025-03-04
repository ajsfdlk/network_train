#pragma once
#include "basic_function.h"
using namespace std;
#define IMAGELENGTH 28
#define IMAGEHEIGHT 28
#define STARTFLOORNUMBER (IMAGELENGTH*IMAGEHEIGHT)  //输入层结点个数
#define ENDFLOORNUMBER 10                           //输出层结点个数
struct image {
	int imageMessage[IMAGELENGTH][IMAGEHEIGHT];
	int imageNum;
};

class network {
public:
	network(vector<int> neuronNum, double studySpeed);//不包含输入层和输出层的值
	void load(string trainImage, string trainLabel);
	void biasesInit();
	void weightsInit();
	void neuronsInit();

	void dumpAll();
	void dumpNeurons();
	void dumpWeights();
	
	void trainInGroup(int epochs, int groupSize, int max);//小批量训练神经网络
	int getRecognitionResult(image* img);//获得神经网络的识别结果	
	void effectEvaluation(string imageFile, string labelFile);//评估训练效果
private:
	vector<double> getDiff(image* img);//获得神经网络对一张图片的代价序列

	double countZ(int l, int j);
	vector<double> countZ(int l);//计算第l列的Z值
	vector<double> weightsMul(int l, vector<double> v);//第l列的权重矩阵与向量v点乘

	double countMatrixNum(int l, int j);//计算aj(l) 第l列的第j个神经元的值（正向）
	void recognitionImage(image* img);//用神经网络识别图片
	//用imageList中从第begin个开始到第end个的一组图片进行训练
	void networkTrain(int begin, int end);
public:
	vector<image> imageList;
private:
	vector<int> neuronsNumList;	
	vector<vector<double>> neurons;//神经元(从第0层开始)
	vector<vector<vector<double>>> weights;
	//权重(从第1层开始，后一层神经元与前一层每个神经元的连接关系)
	//weights[l][j][k]:连接第l列第j个神经元与第l-1列第k个神经元的权重
	vector<vector<double>> biases;//偏置(从第1层开始)
	double studySpeed;
};