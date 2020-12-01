#include "memory.h"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <torch/torch.h>
using namespace std;

torch::Tensor get_data() {
  string line, colname;
  float val;
  torch::Tensor result = torch::zeros({3647, 4}, torch::kFloat32);

  ifstream myFile("cpp_data.csv");
  if (myFile.good()) {
    // std::getline(myFile, line);
    int i = 0;
    while (getline(myFile, line)) {
      std::stringstream ss(line);
      int j = 0;
      while (ss >> val) {
        if (ss.peek() == ',')
          ss.ignore();

        result[i - 1][j] = val;
        if (i == 0) {
          cout << val;
        }
        j++;
      }
      i++;
    }
  }
  return result;
}

int main() {
  // Data is set by row x col [Close, High, Low, Open]
  torch::Tensor data = get_data();
  cout << data[0][0] << "\n" << data[1][0] << endl;
}
