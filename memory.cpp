#include "memory.h"
#include <algorithm>
#include <iostream>
#include <torch/torch.h>
using namespace std;

ReplayMemory::ReplayMemory(int max_mem, int input_dims, int n_actions) {
  mem_cntr = 0;
  idx = 0;
  max_memory = max_mem;
  state_memory = torch::zeros({max_mem, input_dims}, torch::kFloat32);
  action_memory = torch::zeros({max_mem, n_actions}, torch::kInt32);
  reward_memory = torch::zeros(max_mem, torch::kFloat32);
  new_state_memory = torch::zeros({max_mem, input_dims}, torch::kFloat32);
  terminal_memory = torch::zeros(max_mem, torch::kBool);
}

void ReplayMemory::store_transitions(torch::Tensor state, torch::Tensor action,
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

torch::Tensor ReplayMemory::sample_buffer(int batch_size) {
  max_memory = min(mem_cntr, max_memory);
  torch::Tensor min_mem = torch::arange(max_memory, torch::kFloat32);
  batch = torch::multinomial(min_mem, batch_size);
  cout << "BATCH" << batch << endl;
  states = state_memory.index({batch});
  actions = action_memory.index({batch});
  rewards = reward_memory.index({batch});
  states_ = new_state_memory.index({batch});
  dones = terminal_memory.index({batch});
  // Add mapping to return all the tensors
  return states, actions, rewards, states_, dones;
}
