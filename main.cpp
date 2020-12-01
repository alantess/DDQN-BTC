#include "memory.h"
#include <algorithm>
#include <iostream>
#include <torch/torch.h>
using namespace std;

int main() { ReplayMemory memory(3, 1, 4); }
