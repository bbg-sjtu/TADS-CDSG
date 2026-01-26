# TADS-CDSG

This repository contains the code implementation for Trajectory-Attractor Dynamical System (TADS) and Conditional Dynamical System Generation (CDSG).

## File Structure
The implementation of TADS-CDSG is located in the SE3_DS folder.
- **SE3_DS/**: 
  - **dataset/** contains 6 sets of peg-in-hole assembly teaching data and 4 sets of conditional obstacle avoidance teaching data, along with the implementation of the dataset setup.
  - **nn_models/** includes the implementation of invertible neural networks (INN) and conditional INN, as well as the checkpoints for peg-in-hole assembly and conditional obstacle avoidance.
  - **potential_fields/** includes the implementation of the latent space TADS canyon potential field.
  - **trainer/** includes the implementation of the trainer.
  - **utils/** contains several auxiliary tools.
  - **diffeo_train.py**: Unconditional diffeomorphic training and testing.
  - **conditional_diffeo_train.py**: Conditional diffeomorphic training and testing.
  - **diffeo_canyon_sim.py**: The simulation demo of peg-in-hole assembly.

## Demo
The animation of the execution of diffeo_canyon_sim.py is as follows:

![overview](Media/sim_demo.gif)