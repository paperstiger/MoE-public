# Deep RL on indoor navigation problem

DRL is applied on this indoor navigation problem. A car has to move to goal while avoiding obstacle based on either precise localization or distance sensor to the environment.

We set up 4 difficulty levels, single/range goal and sensor/localization.

In order to run the code, you can use
```python
python3 single_goal_problem.py -train
```
to train the model and 
```python
python3 single_goal_problem.py -show
```
to display the results.
This python file can be replaced by range_goal_problem.py, sensor_single_goal_problem.py and sensor_range_goal_problem.py. These files indicate different problem difficulty.

# Customize your problem
You can change the simulation environment by modifying classes in indoor_gym_model.py.
You can tune RL hyperparameters at ppo_util.py.

# Dependencies
pyLib, at [https://gitlab.oit.duke.edu/gt70/pyLib](https://gitlab.oit.duke.edu/gt70/pyLib)