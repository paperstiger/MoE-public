# Benchmark on MoE

In this folder, I test MoE on a list of benchmark problems.
In order to run the code, you have to first download the dataset from [google drive](https://drive.google.com/open?id=1ZyI8RWbgd4CQPJqpQdXXxSUeY_OZjlgY) and unzip into folder *data*.

To visualize the dataset, you have to use command
```python
python show_dataset.py -pcakmean -problem -choice
```
where problem can be chosen from pen, car, drone, dtwo and these four arguments are used throughout this project to select benchmark problems.
The other option choice can be chosen from umap and drawpca which specifies to use umap or PCA to visualize dataset.

To train MoE, simply use function inside train_models.py you can specify benchmark problem in a similar way. See the file for other program options.

To evaluate MoE, run file eval_models.py where you will find detailed description of program arguments.

# data visualization

## show_dataset.py

Show existing dataset.

# generate clusters

## cluster_generation.py

A script to generate several clusters.

# evaluate clusters

## cluster_eval.py

Use metrics to indicate which metric truly reveals discontinuity.

# model training evaluating

## train_models.py

Train a model using data.

## train_hinton_moe.py

Train the MoE as Hinton talked on youtube.

## train_jacob_moe.py

Train MoE using Jacob log probability idea. It turns out to be pretty bad. But it encourage me to warm start it.

## train_mom_net.py

Train the MoM net directly using backpropagation directly.

## eval_models.py

Evaluate all models

# visualization of results

## draw_rollout_result_figures.py

Draw a figure showing all rollout results. This really saves space.

## gen_pen_model_cmp_fig.py

Generate figures comparing the pendulum problem.

# utility functions

## util.py

Yes, utility functions, as the name reveals.

## datanames.py

Record the data file for a few problem.
