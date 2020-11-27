# Apply DDQN on Bitcoin 


Applications of DDQN on the cryptocurrency, Bitcoin.
Goal: Manipulated reward structure to improve DDQN performance

# Environment

  - Single scalar of [Close,High Low, Open , BTC Wallet, USD Wallet]
  - Step: A trade happens at every step: Hold or Purchase (0 - 100%)
  - Reward: Combined rewards ==> δ*r[dense] + r[sparse] , where δ = 1.0 --> 0.0
  - Contains 9 different actions

# Performance
Evaluated performance on bitcoin prices between Nov.7.2020 ~ Nov.25.2020
(Agent starts with $5,000).
![(Performance on test data [1]) Performance](score_plt_test.png "First Test Run")
![(Performance on test data [2]) Performance](score_plt_test_2.png "Second Test Run")



### Installation

Requires [Pytorch](https://pytorch.org/) to run.

```sh
$ cd DDQN-BTC
$ python main.py
```

For training: Run main.py

```sh
$ python main.py -load False
```
- Test performance on price between Nov 7th ~ Nov 25th
- Use "-games" to change the episodes length. 
- Default is 1000 episodes 
```sh
$ python main.py -games 100
```


For unit testing

```sh
$ python test.py
```

License
----

MIT

