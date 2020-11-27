# Apply DDQN on Bitcoin 


Applications of DDQN on the cryptocurrency, Bitcoin.
Goal: Manipulated reward structure to improve DDQN performance

# Environment


  - Single scalar of [Close,High Low, Open , BTC Wallet, USD Wallet]
  - Step: A trade happens at every step: Hold or Purchase (0 - 100%)
  - Reward: Combined rewards ==> δ*r[dense] + r[sparse] , where δ = 1.0 --> 0.0
  - Contains 17 different actions

# Performance
Evaluated performance on bitcoin prices between Nov.7.2020 ~ Nov.25.2020
(Agent starts with $5,000).
![(Performance on test data [1]) Performance](Avg_Earnings.png "First Test Run")

### Installation

Requires [Pytorch](https://pytorch.org/) to run.

```sh
$ cd DDQN-BTC
$ python main.py
```

For training: Run main.py

- Test performance on price between Nov 7th ~ Nov 25th
- Use "-games" to change the episodes length. 
- Default is 1000 episodes 
```sh
$ python main.py ```

```sh
$ python main.py -load=True -games 100
```


For unit testing

```sh
$ python test.py
```
# TODO
- Add more actions 
    - Environment should allow agent to purchase multiple bitcoins instead of 1
- Retrain Agent


License
----

MIT

