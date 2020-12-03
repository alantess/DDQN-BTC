#ifndef MEMORY_H
#define MEMORY_H

#include <torch/torch.h>
class ReplayMemory {
public:
  ReplayMemory(int max_mem, int input_dims, int n_actions);
  void store_transitions(torch::Tensor state, torch::Tensor action,
                         torch::Tensor reward, torch::Tensor state_,
                         torch::Tensor done);
  torch::Tensor sample_buffer(int batch_size = 32);

  torch::Tensor batch, states, actions, rewards, states_, dones;
  int max_memory, idx, mem_cntr;

  torch::Tensor state_memory, action_memory, reward_memory, new_state_memory,
      terminal_memory;
};
#endif
