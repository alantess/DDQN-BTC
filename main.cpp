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
using namespace torch;

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
  torch::Tensor state, state_, reward, reward_calc;
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
    reward_calc =
        torch::tensor(rewards, torch::kFloat32) + (torch::norm(profits) * 0.1);
    rewards = reward_calc.item().to<float>();
    reward = torch::tensor(1.2, torch::kFloat32);

    state_ = get_state();
    // checks for last timestep
    if (timestep == n_steps - 1) {
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
    mapping["state_"] = state_;
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

// Replay Memory
class ReplayBuffer {
public:
  // CONSTRUCTOR
  ReplayBuffer(int a, int b, int c) : max_mem(a), input_dims(b), n_actions(c) {
    idx = 0;
    mem_cntr = 0;
    max_memory = max_mem;
    state_memory = torch::zeros({max_mem, input_dims}, torch::kFloat32);
    action_memory = torch::zeros({max_mem, n_actions}, torch::kInt32);
    reward_memory = torch::zeros(max_mem, torch::kFloat32);
    new_state_memory = torch::zeros({max_mem, input_dims}, torch::kFloat32);
    terminal_memory = torch::zeros(max_mem, torch::kBool);
  }
  // STORE EXPERIENCE
  void store_transitions(torch::Tensor state, torch::Tensor action,
                         torch::Tensor reward, torch::Tensor state_,
                         torch::Tensor done) {
    idx = mem_cntr % max_memory;
    state_memory[idx] = state;
    action_memory[idx] = action;
    reward_memory[idx] = reward;
    new_state_memory[idx] = state_;
    terminal_memory[idx] = done;
    mem_cntr += 1;
  }
  // RETURN EXPERIENCE
  unordered_map<string, torch::Tensor> sample_buffer(int batch_size) {
    max_memory = min(mem_cntr, max_memory);
    min_mem = torch::arange(max_memory, torch::kFloat32);
    batch = torch::multinomial(min_mem, batch_size);
    states = state_memory.index({batch});
    actions = action_memory.index({batch});
    rewards = reward_memory.index({batch});
    states_ = new_state_memory.index({batch});
    dones = terminal_memory.index({batch});
    unordered_map<string, torch::Tensor> mappings;
    mappings["states"] = states;
    mappings["actions"] = actions;
    mappings["rewards"] = rewards;
    mappings["states_"] = states_;
    mappings["dones"] = dones;
    return mappings;
  }
  int max_mem, input_dims, n_actions, idx, mem_cntr, max_memory;
  torch::Tensor states, actions, rewards, states_, dones, state_memory,
      action_memory, reward_memory, new_state_memory, terminal_memory;
  torch::Tensor min_mem, batch;
};

// NETWORK
struct DDQNImpl : nn::Module {
  DDQNImpl(int input_dims, int n_actions, int fc1_dims, int fc2_dims)
      : fc1(input_dims, fc1_dims), fc2(fc1_dims, fc2_dims),
        q(fc2_dims, n_actions) {
    register_module("fc1", fc1);
    register_module("fc2", fc2);
    register_module("q", q);
    torch::Device device(torch::kCUDA);
    this->to(device);
    torch::optim::Adam optimizer(this->parameters(),
                                 torch::optim::AdamOptions(0.00045));
  }
  // Forward Function
  torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(fc1(x));
    x = torch::relu(fc2(x));
    x = q(x);
    return x;
  }
  torch::nn::Linear fc1, fc2, q;
};
TORCH_MODULE(DDQN);
// Agent Class
class Agent {
public:
  Agent(float a, float b, float c, float d, int e, int f, int g, int h, int i,
        int j, int k)
      : lr(a), eps(b), eps_dec(c), gamma(d), input_dims(e), n_actions(f),
        batch_size(g), mem_size(h), replacement(i), fc1_dims(j), fc2_dims(k),
        q_eval(e, f, j, k), q_train(e, f, j, k), memory(h, e, f) {
    update_cntr = 0;
    float eps_min = 0.01;
  }
  // Picks an action
  int pick_action(torch::Tensor obs) {
    float x = random_random();
    if (x > eps) {
      torch::Device device(torch::kCUDA);
      Tensor obs = obs.to(device);
      actions = q_train->forward(obs);
      action = torch::argmax(actions).item().to<int>();
    } else {
      action = sample_actions();
    }
    return action;
  }
  // Stores Experience
  void store_transitions(torch::Tensor state, torch::Tensor action,
                         torch::Tensor reward, torch::Tensor state_,
                         torch::Tensor done) {
    memory.store_transitions(state, action, reward, state, done);
  }
  // Update target networks
  void update_target_net() {
    if (update_cntr % replacement == 0) {

      torch::save(q_train, "q_train_copy.pt");
      torch::load(q_eval, "q_train_copy.pt");
    }
  }
  // Load
  void load() {
    cout << "Loading.." << endl;
    torch::load(q_eval, "q_eval.pt");
    torch::load(q_train, "q_train.pt");
  }
  // Save networks
  void save() {
    cout << "Saving..." << endl;
    torch::save(q_eval, "q_eval.pt");
    torch::save(q_train, "q_train.pt");
  }
  // Learn
  void learn() {

    if (memory.mem_cntr < batch_size) {
      return;
    }
    torch::Device device(torch::kCUDA);
    experience = memory.sample_buffer(batch_size);
    s = experience["states"];
    a = experience["actions"];
    r = experience["rewards"];
    s_ = experience["states_"];
    d = experience["dones"];
    s = s.to(device);
    a = a.to(device);
    r = r.to(device);
    s_ = s_.to(device);
    d = d.to(device);

    q_train->zero_grad();
    update_target_net();
    torch::Tensor indices = torch::arange(batch_size);

    torch::Tensor q_pred = q_train->forward(s) * a;
    q_pred = torch::sum(q_pred, 1);
    torch::Tensor q_next = q_eval->forward(s_);
    torch::Tensor q_train_net = q_train->forward(s_);
    torch::Tensor max_actions = torch::argmax(q_train_net, 1);
    q_next.index({d}) = 0.0;
    torch::Tensor y = r + gamma * q_next.index({indices, max_actions});
    torch::Tensor loss = torch::nn::functional::mse_loss(y, q_pred).to(device);
    loss.backward();

    update_cntr += 1;
    if (eps > eps_min) {
      eps -= eps_dec;
    } else {
      eps = eps_min;
    }
  }

  // Generate a random float (0-1)
  float random_random() {
    std::random_device rd;  // obtain a random number from hardware
    std::mt19937 gen(rd()); // seed the generator
    std::uniform_real_distribution<> distr(0, 1); // define the range
    return distr(gen);                            // generate numbers
  }
  // Generate a radnom action
  int sample_actions() {
    std::random_device rd;  // obtain a random number from hardware
    std::mt19937 gen(rd()); // seed the generator
    std::uniform_int_distribution<> distr(0, 8); // define the range
    return distr(gen);                           // generate numbers
  }
  torch::Tensor s, a, r, s_, d, obs;
  torch::Tensor actions;
  unordered_map<string, torch::Tensor> experience;
  DDQN q_eval, q_train;
  ReplayBuffer memory;
  float lr, eps, eps_dec, eps_min, gamma;
  int action, update_cntr, input_dims, n_actions, batch_size, mem_size,
      replacement, fc1_dims, fc2_dims, random;
};

// MAIN FUNCTION
int main() {

  // Data is set by row x col [Close, High, Low, Open]
  torch::Tensor data = get_data();
  torch::Tensor state, reward, state_, usr_action;
  torch::Tensor s, a, r, s_, d;
  float money = 5000.00;
  unordered_map<string, torch::Tensor> mapping;
  unordered_map<string, torch::Tensor> experience;

  Env env(data, money);
  cout << "Environment created..." << endl;
  Agent agent(0.00045, 1.0, 4e-4, 0.99, 6, 9, 16, 1000000, 1250, 256, 512);
  cout << "Agent created..." << endl;

  cout << "Starting..." << endl;

  torch::optim::Adam train_optimizer(
      agent.q_train->parameters(),
      torch::optim::AdamOptions(2e-4).betas(std::make_tuple(0.5, 0.5)));

  int action;
  for (int i = 0; i < 500; i++) {
    state = env.reset();
    torch::Tensor done = torch::tensor(false, torch::kBool);
    bool finish = done.item<bool>();
    while (!finish) {
      action = env.sample_actions();
      mapping = env.step(action);
      // Assign Variables
      done = mapping["done"];
      state_ = mapping["state_"];
      reward = mapping["reward"];
      usr_action = torch::tensor(action, torch::kInt32);
      agent.store_transitions(state, usr_action, reward, state_, done);
      agent.learn();
      train_optimizer.step();
      if (i + 1 % 25 == 0) {
        torch::save(train_optimizer, "optimizer-checkpoint.pt");
      }
      finish = done.item<bool>();
    }
    cout << "Episode " << i << "\t" << mapping["info"] << "\tReward" << reward
         << endl;
  } // End of loop
}
