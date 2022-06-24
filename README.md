# Robust Soft Actor Critic

I designed new Robust Deep RL with a Soft Actor-Critic approach with adversarial perturbation on state observations. My work is based on SA-MDP, which is proposed by Zhang et al. (2020).

### SA-MDP(State Adversarial-MDP)

SA-MDP assumes that the fixed-adversarial attack is the situation of the worst-case with the most minimized Q value following equations, and Zhang et al. (2020) newly define it as a SA-MDP.

	$\tilde{V}_{\pi \circ \nu}(s) = \sum\limits_{a\in\nu(s)} \pi(a|\nu(s)) \sum\limits_{s'\in S} p(s'|s,a) \left[ R(s,a,s')+\gamma\tilde{V}_{\pi \circ \nu}(s') \right]$

	$\tilde{Q}_{\pi \circ \nu}(s) = \sum\limits_{s'\in S} p(s'|s,a) \left[R(s,a,s')+\gamma     \sum\limits_{a''\in A}\pi(a'|\nu(s'))) \tilde{Q}_{\pi \circ \nu}(s',a')\right]$

$\tilde{V}_{\pi \circ \nu^*}(s) = \min_{\nu}  \tilde{V}_{\pi \circ \nu}(s), 	\tilde{Q}_{\pi \circ \nu^*}(s,a) = \min_{\nu}  \tilde{Q}_{\pi \circ \nu}(s,a)$

### SA-SAC Regularizer

 $R_{SAC}(\theta_{\pi},\bar s_{i}):= \sum_{i}^{}\max_{\bar{s_i}\in B_p(s_t,\epsilon_t)}||\pi_{\theta_\pi}(s_i)-\pi_{\theta_\pi}(\bar{s_i})||_2$

### SA-SAC

In our work,  we need to solve a minimax problem: **minimizing the policy loss for a worst case**

- object function

$\nabla_{\theta} \frac{1}{|B|}\sum_{s\in B}^{}(\min_{i=1,2}Q_{\phi_i}(s,\tilde{a}_{\theta}(s))-\alpha \log\pi_\theta(\tilde{a}_{\theta}(s)|s)-\kappa_{SAC}\nabla_{\theta_\pi}\bar{R}_{SAC})$

# Codes

I designed **Robust Deep RL with a soft actor critic approach in discrete action space**. I tested **SA-SAC** in a several **atari gym** environments.

# Results

![Untitled](Readme%20aadfcebd405747f7838ef4003bc87d05/Untitled.png)

![Untitled](Readme%20aadfcebd405747f7838ef4003bc87d05/Untitled%201.png)

# Train SA-SAC agent

- Make new three directories `saved_models`, `vidoes` and `Logs` .
- Before you start training, set `n_steps`, `memory_size`, `train_start`, `reg_train_start` … at the `config01.json` file.
- `n_steps` : total nubmer of steps you want to train.
- `memory_size`: buffer memory size
- `train_start:` number of steps when training begins.
- `reg_train_start`: number of steps when training with SA-Regularizer begins.

### [train.py](http://train.py) (train vanilla SAC)

```python
train.py 
	--config=config01.json(default)
	--new=1(default) # set 0 when you load pretrained models
  --game=BeamRider(default) # set any atari game environment 
```

- example:  `python [train.py](http://train.py)` ,  `python [train.py](http://train.py) —game=Assault`

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

### reference

**[Robust Deep Reinforcement Learning against Adversarial ...**https://arxiv.org › cs](https://arxiv.org/abs/2003.08938)