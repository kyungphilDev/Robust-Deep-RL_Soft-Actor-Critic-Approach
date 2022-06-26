# Robust Soft Actor Critic

I designed new Robust Deep RL with a Soft Actor-Critic approach with adversarial perturbation on state observations. My work is based on SA-MDP, which is proposed by Zhang et al. (2020). For more detailed explanation, please check attached pdf file. **[2022 Spring Semester, Personal Project Research _Kyungphil Park](https://github.com/kyungphilDev/Robust-Deep-RL_Soft-Actor-Critic-Approach/blob/master/Robust%20Deep%20Reinforcement%20Learning%20for%20Soft-Actor-Critic%20against%20Adversarial%20Perturbation%20.pdf)

### SA-MDP(State Adversarial-MDP)

SA-MDP assumes that the fixed-adversarial attack is the situation of the worst-case with the most minimized Q value following equations, and Zhang et al. (2020) newly define it as a SA-MDP. **[Zhang et al. (2020)](https://arxiv.org/abs/2003.08938)

![1](https://user-images.githubusercontent.com/80669616/175530915-d2a208b0-0452-401b-bea1-40b4d1266b08.jpg)


### SA-SAC Regularizer

![3](https://user-images.githubusercontent.com/80669616/175531143-cc6c646c-ce36-43c5-a6b8-202b21d9a9a8.jpg)


### SA-SAC

In our work,  we need to solve a minimax problem: **minimizing the policy loss for a worst case**

- object function

![4](https://user-images.githubusercontent.com/80669616/175530884-1fde8fb9-9828-4ef1-a04f-cdb882c32f41.jpg)

# Codes

I designed **Robust Deep RL with a soft actor critic approach in discrete action space**. I tested **SA-SAC** in a several **atari gym** environments.
SAC codes are based on the **[bernomone's github codes](https://github.com/bernomone/SoftActorCritic_ReinforcementLearning).

# Train SA-SAC agent
At first, make new three directories `saved_models`, `vidoes` and `Logs`.
- Before you start training, set `n_steps`, `memory_size`, `train_start`, `reg_train_start` … at the `config01.json` file.
- `n_steps` : total nubmer of steps you want to train.
- `memory_size`: buffer memory size
- `train_start:` number of steps when training begins.
- `reg_train_start`: number of steps when training with SA-Regularizer begins.

### train.py (train vanilla SAC)

```python
train.py 
	--config=config01.json(default)
	--new=1(default) # set 0 when you load pretrained models
 	--game=BeamRider(default) # set any atari game environment 
```

- example:  `python train.py` ,  `python [train.py](http://train.py) —game=Assault`

### robust_train.py (train SA-SAC)

```python
robust_train.py 
	--config=config01.json(default)
	--new=1(default) # set 0 when you load pretrained models
 	--game=BeamRider(default) # set any atari game environment 
```

- example: `python robust_train.py` , `python robust_[train.py](http://train.py) —game=Assault`

### generate_match_video.py

- render **atari game video** with your **trained models**.

```python
generate_match_video.py
	--config=config01.json(default)
	--seed=0(default)
  	--game=BeamRider(default) # set any atari game environment 
  	--random=False(default) # set 1 when you want to test random action.
```

- example: `python generate_match_video.py`, `python generate_match_video[.py](http://train.py) —game=Assault --random=1`

### PGD_generate_video.py

(+ **PGD attack(adversarial perturbation on state observation)**

- render **atari game video** with your **trained models**

```python
PGD_generate_video.py
	--config=config01.json(default)
	--seed=0(default)
	--game=BeamRider(default) # set any atari game environment 
  	--steps=10(default) # set PGD attack steps number.
```

- example: `python PGD_generate_video.py`, `python PGD_generate_video[.py](http://train.py) —game=Assault`

### evalulation.py

- test trained models for several episodes.

```python
evalulation.py
	--config=config01.json(default)
	--seed=0(default)
  	--game=BeamRider(default) # set any atari game environment 
  	--iter=10(default) # set iteration number(tot episode number).
```

- example: `python evalulation.py`, `python evalulation[.py](http://train.py) —game=Assault —iter=30`

### pgd_evalulation.py

(+ **PGD attack(adversarial perturbation on state observation)**

- test trained models for several episodes.

```python
pgd_evalulation.py
	--config=config01.json(default)
	--seed=0(default)
  	--game=BeamRider(default) # set any atari game environment 
  	--iter=10(default) # set iteration number(tot episode number).
```

- example: `python pgd_evalulation.py`, `python pgd_evalulation[.py](http://train.py) —game=Assault —iter=30`

# Results

![Untitled 1](https://user-images.githubusercontent.com/80669616/175530312-7bdc026b-2c51-4c41-ac9e-1eb829c41e66.png)
![Untitled](https://user-images.githubusercontent.com/80669616/175530367-1fb75530-f419-404c-8368-665ff1a3836f.png)


# References

[1] [Zhang et al. (2020)](https://arxiv.org/abs/2003.08938)
[2] [bernomone's github codes](https://github.com/bernomone/SoftActorCritic_ReinforcementLearning)
