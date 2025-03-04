#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "network.h"
using namespace std;

int main()
{
	network myNet({30},2);
	myNet.load("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
	//myNet.dumpAll();
	myNet.effectEvaluation("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");
	myNet.trainInGroup(5, 10,60000);
	//myNet.dumpAll();
	myNet.effectEvaluation("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");
	return 0;
}