 # Cmpe591 Deep Learning for Robotics Project 2: 

## How to run:
homework4.py contains the data collecting and training script in the __main__ block using the pre-implemented CNP class.

- Data collecting: new data can be collected by setting the use_existing_trajectories variable false. Otherwise it uses the previously collected data.
- Training: New model can be trained by setting the train_new_model variable true. Otherwise it uses the previously trained best model. The training script is in the "if train_new_model:" block.
- Testing and plotting: The model is tested on 20 novel trajectories by doing 5 tests on each trajectory, totaling 100 tests. The mean error and the standard deviation for the 100 tests are plotted below.

## Error Plots:

![CNMP_error_bar_plot](https://github.com/user-attachments/assets/5ba77559-368f-4873-8063-38e4cb9b7406)
