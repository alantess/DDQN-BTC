#include "memory.h"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <torch/torch.h>
#include <unordered_map>

using namespace std;

// CSV to Torch Tensor
torch::Tensor get_data() {
  string line;
  float val;
  torch::Tensor result = torch::zeros({3647, 4}, torch::kFloat32);
  // Read file
  ifstream myFile("cpp_data.csv");
  if (myFile.good()) {
    int i = 0;
    while (getline(myFile, line)) {
      std::stringstream ss(line);
      int j = 0;
      while (ss >> val) {
        if (ss.peek() == ',')
          ss.ignore();

        result[i - 1][j] = val;

        j++;
      }
      i++;
    }
  }
  return result;
};

class Env {
public:
  // Variable user gets back
  torch::Tensor state, state_, reward;
  torch::Tensor data, profits, action_space, observation_space, btc_price, done,
      usr_action;
  float investment, prev_holdings, new_holdings, rewards, usd_wallet,
      btc_wallet, reward_dec, total;
  int size, n_steps, timestep, n_headers;

  Env(torch::Tensor x, float y) : data(x), investment(y) {
    size = 3647;
    n_steps = size;
    n_headers = 4;
    timestep = 0;
    usd_wallet = investment;
    btc_wallet = 0;
    reward_dec = 1.0;
    total = 0;
    profits = torch::empty(size, torch::kFloat32);
    action_space = torch::arange(9);
    observation_space = torch::zeros(n_headers + 2, torch::kFloat32);
    reset();
  }
  // Reset the environment
  torch::Tensor reset() {
    timestep = 0;
    torch::Tensor profits = torch::empty(size, torch::kFloat32);
    btc_wallet = 0;
    usd_wallet = investment;
    get_price();
    reward_dec = (reward_dec > 0) ? reward_dec : 0;
    total = 0;
    state = get_state();
    return state;
  }
  // Take a step in the environment
  unordered_map<string, torch::Tensor> step(int action) {
    assert(action >= 0 && action < 9);
    get_price();
    rewards = 0.0;
    reward = torch::tensor(0, torch::kFloat32);
    prev_holdings = btc_wallet + usd_wallet;
    // Perform Action and update wallets
    action_set(action);
    update_btc_wallet();
    // Increment timestep
    timestep += 1;
    // Rewards calc
    new_holdings = btc_wallet + usd_wallet;
    total = new_holdings;
    profits[timestep - 1] = new_holdings;
    rewards = ((new_holdings - prev_holdings) * reward_dec) * 0.5;
    reward = torch::tensor(rewards, torch::kFloat32) + (norm(profits) * 0.1);

    // s'
    state_ = get_state();
    // checks for last timestep
    if (timestep == n_steps) {
      done = torch::tensor(true, torch::kBool);
    } else {
      done = torch::tensor(false, torch::kBool);
    }

    // Actions to tensor
    usr_action = torch::tensor(action);
    // info
    torch::Tensor info = torch::tensor(total);

    // return multiple values
    unordered_map<string, torch::Tensor> mapping;
    mapping["state"] = state;
    mapping["reward"] = reward;
    mapping["done"] = done;
    mapping["info"] = info;
    return mapping;
  }
  // Choose ranndom actions
  int sample_actions() {
    std::random_device rd;  // obtain a random number from hardware
    std::mt19937 gen(rd()); // seed the generator
    std::uniform_int_distribution<> distr(0, 8); // define the range
    return distr(gen);                           // generate numbers
  }

private:
  // Frobenius norm
  template <class tensor> tensor norm(tensor z) {
    tensor y = torch::tensor(0, torch::kFloat32);
    for (int i = 0; i < z.sizes()[0]; i++) {
      y += torch::abs(torch::pow(z[i], 2));
    }
    return torch::sqrt(y);
  }

  // Actions set descriptions
  void action_set(int action) {
    switch (action) {
    case 0:
      break;
    case 1:
      buy_or_sell(true, 1.0);
      break;
    case 2:
      buy_or_sell(false, 1.0);
      break;
    case 3:
      buy_or_sell(true, 0.75);
      break;
    case 4:
      buy_or_sell(false, 0.75);
      break;
    case 5:
      buy_or_sell(true, 0.5);
      break;
    case 6:
      buy_or_sell(false, 0.5);
      break;
    case 7:
      buy_or_sell(true, 0.25);
      break;
    case 8:
      buy_or_sell(false, 0.25);
      break;
    }
  }

  // Buys or Sells Crypto
  void buy_or_sell(bool purchase, float percentage) {
    // Convert Tensor to float
    float price = btc_price.item<float>();
    // Calculate amount for buying or selling
    float amount = price * percentage;
    if (purchase) {
      if (usd_wallet > amount) {
        usd_wallet -= amount;
        btc_wallet += amount;
      }
    } else {
      if (btc_wallet >= amount) {
        btc_wallet -= amount;
        usd_wallet += amount;
      }
    }
  }

  // Updates bitcoin wallet
  void update_btc_wallet() {
    torch::Tensor btc_tensor_val = data[timestep + 1][0] / btc_price;
    btc_wallet *= btc_tensor_val.item<float>();
  }

  // Retrieve btc price
  void get_price() { btc_price = data[timestep][0]; }

  // Gets the state
  torch::Tensor get_state() {
    state = observation_space;
    for (int i = 0; i < n_headers; i++) {
      state[i] = data[timestep][i];
    }
    state[4] = btc_wallet;
    state[5] = usd_wallet;
    return state;
  }
};

int main() {
  // Data is set by row x col [Close, High, Low, Open]
  torch::Tensor data = get_data();
  torch::Tensor state;
  float money = 5000.00;

  Env env(data, money);

  int action;
  unordered_map<string, torch::Tensor> mapping;
  // Example loop
  for (int k = 0; k < 10; k++) {
    cout << "Step " << k << endl;
    action = env.sample_actions();
    cout << "Action Taken: " << action << endl;
    mapping = env.step(action);
    cout << "STate_" << mapping["state"] << "\nInfo" << mapping["info"]
         << "\nDone" << mapping["done"] << "\nREWARD" << mapping["reward"]
         << endl;
  }
}
