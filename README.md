# CartPole-DQN
Deep Q-Network algorithm implemented on openai gym cartPole environment

Special thanks to @sentdex, since I have been following his tutorial for months, which is how I began and got familiar with programming related to Artificial Intelligence and successfully wrote an my own dqn-code to date

# Requirments
Python 3.6

tensorflow-gpu

gym

# Source

https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf

# Just in case
step1: Download python3 from "https://www.python.org/"

step2: Install python by following prompt

step3: Open cmd, enter the installation location, which usually exists in C:\Users\XXX\AppData\Local\Python\Python36\Scripts 

(win+r >> type: 'cmd' >> type: 'cd C:\Users\XXX\AppData\Local\Python\Python36\Scripts')

step4: Type: 'pip install tensorflow gym==0.12.4'

(if you have nvidia gpu, add 'pip install tensorflow-gpu')

step5: Run 'train_model.py' to train your own model that balances the cartpole, or run 'test_model.py' or 'test_model_with_no_episodes.py' to test the pretrained models given.

# Addition

Among those models, you might find some versions of models missing: 'v4', 'v9', 'v10' and 'v12'. In fact, due to some reasons, those models yielded terrible results, and thus are removed from the repository

# Special Thanks

Sentdex
