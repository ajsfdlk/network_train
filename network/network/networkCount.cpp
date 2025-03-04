#include "network.h"

double network::countZ(int l, int j)
{
	double a = 0;
	if (l < 1 || l >= neurons.size())
		return 0;
	if (j < 0 || j >= neurons[l].size())
		return 0;

	for (int i = 0; i < neurons[l - 1].size(); i++)
		a += weights[l - 1][j][i] * neurons[l - 1][i];

	a += biases[l - 1][j];
	return a;
}

vector<double> network::countZ(int l)
{
	vector<double> res;
	if (l < 1 || l >= neurons.size())
		return res;
	for (int i = 0; i < neurons[l].size(); i++)
		res.push_back(countZ(l, i));
	return res;
}

double network::countMatrixNum(int l, int j)
{
	return sigmoid(countZ(l, j));
}

vector<double> network::weightsMul(int l, vector<double> v)//w(l)T*v
{
	vector<double> res;
	if (l < 1 || l >= neurons.size())
		return res;
	l--;
	vector<vector<double>> m = transposeInPlace(weights[l]);
	for (int i = 0; i < m.size(); i++)
	{
		double res_mid=0;
		for (int j = 0; j < m[i].size(); j++)
			res_mid += m[i][j] * v[j];
		res.push_back(res_mid);
	}
	return res;
}