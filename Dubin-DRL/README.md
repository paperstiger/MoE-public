# Deep RL on Dubin car problem

In this folder, DRL is applied on a dubin car problem.

To run this example, simply type 
```python
python3 dubinCar.py -train -small
```
to train the model and 
```python
python3 dubinCar.py -show -small
```
to show the results.
There are many more options, please see the file for detailed documentation of available options.

# Modification to the environment
In file externalmodel.py the class DubinCarEnv is defined. You could modify code there to adjust the model.
You can tune RL hyperparameters at ppo_util.py.

# Dependencies
pyLib, at [https://gitlab.oit.duke.edu/gt70/pyLib](https://gitlab.oit.duke.edu/gt70/pyLib)