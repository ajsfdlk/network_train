#include "network.h"

void network::networkTrain(int begin, int end)
{
	if (begin<0 || end>imageList.size() - 1||begin>end)
		return;
	vector<vector<vector<double>>> weightsChangeBuffer;//暂时存储一次反向传播得到的梯度
	vector<vector<double>> biasesBuffer;//偏置值梯度暂存
	for (int i = 0; i < weights.size(); i++)
	{
		vector<vector<double>> list_mid;
		for (int j = 0; j < weights[i].size(); j++)
			list_mid.push_back(vector<double>(weights[i][j].size(), 0));
		weightsChangeBuffer.push_back(list_mid);
	}
	for (int i = 0; i < biases.size(); i++)
		biasesBuffer.push_back(vector<double>(biases[i].size(), 0));

	for (int i = begin; i <= end; i++)
	{
		image* img = &imageList[i];

		int l = neurons.size()-1;
		vector<double> cost = torchMul(getDiff(img), sigmoid_prime_vec(countZ(l)));
		//获取一次识别后的输出层梯度:δ_L=∇aC⊙σ′(z_L)

		for (; l > 0; l--)//一层层向前传播
		{
			//计算偏置值修改量：∂C/∂b(l)_j=δ(l)_j
			for (int j = 0; j < cost.size(); j++)
				biasesBuffer[l-1][j] += cost[j];
			//计算权重修改量：∂C/∂w(l)_jk=a(l-1)_j*δ(l)_k
			for (int j = 0; j < weightsChangeBuffer[l-1].size(); j++)
			{
				for (int k = 0; k < weightsChangeBuffer[l - 1][j].size(); k++)
					weightsChangeBuffer[l - 1][j][k] += neurons[l-1][k] * cost[j];
			}

			vector<double> cost_now= torchMul(weightsMul(l,cost), 
				sigmoid_prime_vec(countZ(l - 1)));//第L-1层的梯度
			cost = cost_now;
		}
	}

	int groupNum = end - begin + 1;
	for (int l = 0; l < weights.size(); l++)
	{
		for (int j = 0; j < weights[l].size(); j++)
		{
			for (int k = 0; k < weights[l][j].size(); k++)
				weights[l][j][k] -= studySpeed * weightsChangeBuffer[l][j][k] / groupNum;
		}
	}
	for (int l = 0; l < biases.size(); l++)
	{
		for (int j = 0; j < biases[l].size(); j++)
			biases[l][j] -= studySpeed * biasesBuffer[l][j] / groupNum;
	}
	//dumpWeights();
}

void network::trainInGroup(int epochs, int groupSize,int max)
{
	if (groupSize <= 0)
		return;
	if (max > imageList.size())
		max = imageList.size();
	if (groupSize > max)
		groupSize = max;
	random_shuffle(imageList.begin(), imageList.end());//将图片顺序打乱
	for (int i = 0; i < epochs; i++)
	{
		clock_t t1 = time(NULL);
		cout << "第" << i+1 << "轮:" << endl;
		int start = 0, end = groupSize-1;
		for (int j = 0; j < max / groupSize; j++)
		{
			networkTrain(start, end);
			start += groupSize;
			end += groupSize;
			cout << "#";
		}
		cout << endl;
		clock_t t2 = time(NULL);
		cout << "time:" << t2 - t1 << endl;
	}
}

void network::effectEvaluation(string imageFile, string labelFile)
{
	vector<image> image_list;//读取测试数据集
	vector<vector<double>>images;
	vector<double>labels;
	read_Mnist_Images(imageFile, images);
	read_Mnist_Label(labelFile, labels);
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
		image_list.push_back(im_mid);
	}
	cout << "read dataset finish\nget image num:" << image_list.size() << endl;

	double right_num = 0;
	for (int i = 0; i < image_list.size(); i++)
	{
		int result = getRecognitionResult(&image_list[i]);
		if (result == image_list[i].imageNum)
			right_num++;
	}
	cout << "验证：" << right_num << "/" << image_list.size() << " 正确率："
		<< (right_num / image_list.size()) * 100 << "%" << endl;
}