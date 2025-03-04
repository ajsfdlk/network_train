#include "basic_function.h"
int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void read_Mnist_Label(string filename, vector<double>& labels)
{
	ifstream file(filename, ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_images, sizeof(number_of_images));
		magic_number = ReverseInt(magic_number);
		number_of_images = ReverseInt(number_of_images);
		cout << "magic number = " << magic_number << endl;
		cout << "number of images = " << number_of_images << endl;


		for (int i = 0; i < number_of_images; i++)
		{
			unsigned char label = 0;
			file.read((char*)&label, sizeof(label));
			labels.push_back((double)label);
		}

	}
	else
		cout << "fail to open:" << filename << endl;
}

void read_Mnist_Images(string filename, vector<vector<double>>& images)
{
	ifstream file(filename, ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		unsigned char label;
		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_images, sizeof(number_of_images));
		file.read((char*)&n_rows, sizeof(n_rows));
		file.read((char*)&n_cols, sizeof(n_cols));
		magic_number = ReverseInt(magic_number);
		number_of_images = ReverseInt(number_of_images);
		n_rows = ReverseInt(n_rows);
		n_cols = ReverseInt(n_cols);

		cout << "magic number = " << magic_number << endl;
		cout << "number of images = " << number_of_images << endl;
		cout << "rows = " << n_rows << endl;
		cout << "cols = " << n_cols << endl;

		for (int i = 0; i < number_of_images; i++)
		{
			vector<double>tp;
			for (int r = 0; r < n_rows; r++)
			{
				for (int c = 0; c < n_cols; c++)
				{
					unsigned char image = 0;
					file.read((char*)&image, sizeof(image));
					tp.push_back(image);
				}
			}
			images.push_back(tp);
		}
	}
	else
		cout << "fail to open:" << filename << endl;
}

double sigmoid(double z)
{
	return 1 / (1 + exp(-z));
}

double sigmoid_prime(double z)
{
	return exp(-z) / ((1 + exp(-z)) * (1 + exp(-z)));
}

void dump_double_list(vector<double> list)
{
	cout << "size:" << list.size() << endl;
	for (auto iter : list)
		cout << iter << " ";
	cout << endl;
}

void vectorMulti(vector<double>& list, double t)
{
	for (int i = 0; i < list.size(); i++)
		list[i] *= t;
}

vector<double> torchMul(vector<double> t1, vector<double> t2)
{
	vector<double> res;
	if (t1.size() != t2.size())
		return res;

	for (int i = 0; i < t1.size(); i++)
		res.push_back(t1[i] * t2[i]);
	return res;
}

vector<vector<double>> transposeInPlace(vector<vector<double>> m) 
{
	vector<vector<double>> array;
	vector<double> temparay;
	for (int i = 0; i < m[0].size(); ++i)// m*n ά����
	{
		for (int j = 0; j < m.size(); ++j) {
			temparay.push_back(m[j][i]);
		}
		array.push_back(temparay);
		temparay.erase(temparay.begin(), temparay.end());
	}
	return array;
}

vector<double> sigmoid_prime_vec(vector<double> z)
{
	for (int i = 0; i < z.size(); i++)
		z[i] = sigmoid_prime(z[i]);
	return z;
}