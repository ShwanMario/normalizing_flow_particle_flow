# normalizing_flow_particle_flow

This code is used for comparing the convergence rate of particle flow with normalizing flow, we evaluate their performance on MNIST dataset.

# Running the experiments
This code allows you to train vae with particle flow or normalizing flow on MNIST dataset. To train the model, run the following commands.
## Trainning VAE with particle flow
```
python train_particle_flow.py
```
## Training VAE with normalizing flow
```
python main_train.py
```

See [the training file for particle flow](https://github.com/ShwanMario/IWAE/blob/master/Importance_Weighted_Autoencoders-master/MNIST/script/main_train.py) and [the training flie for normalizing flow](https://github.com/ShwanMario/IWAE/blob/master/Importance_Weighted_Autoencoders-master/MNIST/script/main_test.py) for more configurations.
