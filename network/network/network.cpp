#include "network.h"
void network::biasesInit()
{
	biases.clear();
	// 创建随机数生成器
	std::random_device rd;  // 获取随机数种子
	std::mt19937 gen(rd()); // 使用梅森旋转算法生成随机数
	std::normal_distribution<> d(0.0, 1.0); // 标准正态分布，均值0，标准差1
	for (int i = 1; i < this->neuronsNumList.size(); i++)
	{
		vector<double> biases_mid;
		for (int j = 0; j < this->neuronsNumList[i]; j++)
			biases_mid.push_back(d(gen));
		biases.push_back(biases_mid);
	}
}

void network::neuronsInit()
{
	neurons.clear();
	for (int i = 0; i < this->neuronsNumList.size(); i++)
		neurons.push_back(vector<double>(this->neuronsNumList[i], 0));
		//初始化神经元结点，初始值为0
}

void network::weightsInit()
{
	weights.clear();

	// 创建随机数生成器
	std::random_device rd;  // 获取随机数种子
	std::mt19937 gen(rd()); // 使用梅森旋转算法生成随机数
	std::normal_distribution<> d(0.0, 1.0); // 标准正态分布，均值0，标准差1
	int lastFloorNum = STARTFLOORNUMBER;
	for (int i = 1; i < this->neuronsNumList.size(); i++)
	{
		vector<vector<double>> weight_mid1;

		for (int j = 0; j < this->neuronsNumList[i]; j++)//初始化权重列表
		{
			vector<double> weight_mid;
			for (int k = 0; k < lastFloorNum; k++)
				weight_mid.push_back(d(gen));
			weight_mid1.push_back(weight_mid);
		}

		lastFloorNum = this->neuronsNumList[i];
		weights.push_back(weight_mid1);
	}
}

network::network(vector<int> neuronNum, double studySpeed)
{
	this->studySpeed = studySpeed;

	this->neuronsNumList.push_back(STARTFLOORNUMBER);//初始化神经网络结构表
	this->neuronsNumList.insert(this->neuronsNumList.end(),
		neuronNum.begin(), neuronNum.end());
	this->neuronsNumList.push_back(ENDFLOORNUMBER);

	neuronsInit();
	weightsInit();
	biasesInit();
}

void network::load(string trainImage, string trainLabel)
{
	imageList.clear();
	vector<vector<double>>images;
	vector<double>labels;
	read_Mnist_Images(trainImage, images);
	read_Mnist_Label(trainLabel, labels);
	cout << "start to read" << endl;
	for (int i = 0; i < images.size(); i++)
	{
		image im_mid;
		for (int j = 0; j < IMAGELENGTH; j++)
		{
			for (int k = 0; k < IMAGEHEIGHT; k++)
			{
				im_mid.imageMessage[j][k] = images[i][j * IMAGELENGTH + k];
			}
		}
		im_mid.imageNum = labels[i];
		imageList.push_back(im_mid);
	}
	cout << "read dataset finish\nget image num:" << imageList.size() << endl;
}

void network::recognitionImage(image* img)
{
	if (!img)
		return;
	int* pixel = &(img->imageMessage[0][0]);
	for (int i = 0; i < STARTFLOORNUMBER; i++)//模型接收图片数据
	{
		neurons[0][i] = sigmoid(*pixel);
		pixel++;
	}
	for (int i = 1; i < neurons.size(); i++)//从前向后传播
	{
		for (int j = 0; j < neurons[i].size(); j++)
			neurons[i][j] = countMatrixNum(i, j);
	}
	//dumpNeurons();
}

int network::getRecognitionResult(image* img)
{
	if (!img)
		return -1;
	recognitionImage(img);
	double maxNeurons = 0;
	int result = 1;
	for (int i = 0; i < neurons[neurons.size() - 1].size(); i++)
	{
		if (neurons[neurons.size() - 1][i] > maxNeurons)
		{
			result = i;
			maxNeurons = neurons[neurons.size() - 1][i];
		}
	}
	return result;
}

vector<double> network::getDiff(image* img)
{
	vector<double> diff;
	if (!img)
		return diff;
	recognitionImage(img);
	int result = img->imageNum;
	for (int i = 0; i < ENDFLOORNUMBER; i++)
		diff.push_back((i == result) ? 1 : 0);
	for (int i = 0; i < diff.size(); i++)
		diff[i] = neurons[neurons.size() - 1][i]- diff[i];//y-a
	return diff;
}

void network::dumpNeurons()
{
	cout << "\nneurons:" << endl;
	for (int i = 0; i < neurons.size(); i++)
	{
		cout << "column:" << i << endl;
		for (int j = 0; j < neurons[i].size(); j++)
			cout << neurons[i][j] << " ";
		cout << endl;
	}
}

void network::dumpWeights()
{
	cout << "\nweights:" << endl;
	for (int i = 0; i < weights.size(); i++)
	{
		cout << "column:" << i + 1 << endl;
		for (int j = 0; j < weights[i].size(); j++)
		{
			cout << "id:" << j << endl;
			for (int k = 0; k < weights[i][j].size(); k++)
				cout << weights[i][j][k] << " ";
			cout << endl;
		}
	}
}

void network::dumpAll()
{
	dumpNeurons();

	dumpWeights();

	cout << "\nbiases:" << endl;
	for (int i = 0; i < biases.size(); i++)
	{
		cout << "column:" << i+1 << endl;
		for (int j = 0; j < biases[i].size(); j++)
			cout << biases[i][j] << " ";
		cout << endl;
	}
}