# SoftActorCritic_ReinforcementLearning

This is my implementation of the discrete Soft Actor Critic algorithm for Reinforcement Learning. More details about the algorithm can be found here:

- Christodoulou, Petros. "Soft actor-critic for discrete action settings." arXiv preprint arXiv:1910.07207 (2019).

This implementation works with all the Open AI Gym Environments having a discrete action set, but right now it plays **Space Invaders**.

# How to use it?

The requirements for the repo to work are in env.yml (if you use conda: conda env create -f env.yml).

First, edit the __config01.json__ file (or create a new one), with the metaparameters and variables used for the training:

- configId: string identifier of the traning, 
- env_parameters:
    - screen_size: size of the video game screen,
    - frame_skip: frame skip value, 
    - seed_value: random number generators' seed 
- training_parameters:
    - n_episodes: number of episodes to play during training
    - t_tot_cut: maximum number of moves to play in each episode
    - batch_size: number of previous moves to use during a training step

- agent_parameters:
    - gamma: discount rate
    - lr_Q: learning rate of the Q-function
    - lr_pi: learning rate of the policy
    - lr_alpha: learning rate of the temperature
    - alpha: value of the temperature. Set it to "auto" if you want it to be learnt during training
    - tau: update rate of the Q target functions
    - entropy_rate: constant to be put in front of the target entropy if alpha="auto". The lower, the less random the moves will be at the end.
    - h_dim: hidden units of the Q-functions flat layers
    - h_mu_dim: hidden units of the policy flat layers


The run "python train.py" to start training. The training curve will be saved into the "train_figs" folder every 100 episodes, while the model parameters will be stored in "saved_models". <br>

When the training is over, launch "python best_of_100_episodes.py" to have the best episode reward out of a sample of 100. Best score on the Open AI Leader Board (https://github.com/openai/gym/wiki/Leaderboard) is 3454. Here, we should get around 1800. <br>

To generate a small movie of a match use instead "python generate_match_video.py". There is an example in the __/video__ directory.


https://user-images.githubusercontent.com/12211213/148438233-441f4ea5-2513-41fd-93da-e855384f13ef.mp4


