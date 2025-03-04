#pragma once
#include "basic_function.h"
using namespace std;
#define IMAGELENGTH 28
#define IMAGEHEIGHT 28
#define STARTFLOORNUMBER (IMAGELENGTH*IMAGEHEIGHT)  //����������
#define ENDFLOORNUMBER 10                           //����������
struct image {
	int imageMessage[IMAGELENGTH][IMAGEHEIGHT];
	int imageNum;
};

class network {
public:
	network(vector<int> neuronNum, double studySpeed);//������������������ֵ
	void load(string trainImage, string trainLabel);
	void biasesInit();
	void weightsInit();
	void neuronsInit();

	void dumpAll();
	void dumpNeurons();
	void dumpWeights();
	
	void trainInGroup(int epochs, int groupSize, int max);//С����ѵ��������
	int getRecognitionResult(image* img);//����������ʶ����	
	void effectEvaluation(string imageFile, string labelFile);//����ѵ��Ч��
private:
	vector<double> getDiff(image* img);//����������һ��ͼƬ�Ĵ�������

	double countZ(int l, int j);
	vector<double> countZ(int l);//�����l�е�Zֵ
	vector<double> weightsMul(int l, vector<double> v);//��l�е�Ȩ�ؾ���������v���

	double countMatrixNum(int l, int j);//����aj(l) ��l�еĵ�j����Ԫ��ֵ������
	void recognitionImage(image* img);//��������ʶ��ͼƬ
	//��imageList�дӵ�begin����ʼ����end����һ��ͼƬ����ѵ��
	void networkTrain(int begin, int end);
public:
	vector<image> imageList;
private:
	vector<int> neuronsNumList;	
	vector<vector<double>> neurons;//��Ԫ(�ӵ�0�㿪ʼ)
	vector<vector<vector<double>>> weights;
	//Ȩ��(�ӵ�1�㿪ʼ����һ����Ԫ��ǰһ��ÿ����Ԫ�����ӹ�ϵ)
	//weights[l][j][k]:���ӵ�l�е�j����Ԫ���l-1�е�k����Ԫ��Ȩ��
	vector<vector<double>> biases;//ƫ��(�ӵ�1�㿪ʼ)
	double studySpeed;
};