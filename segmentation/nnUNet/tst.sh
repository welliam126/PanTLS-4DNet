# #!/bin/bash

# # Set the CUDA_VISIBLE_DEVICES to 5(GPU device index)
# export CUDA_VISIBLE_DEVICES=5

# # Define a list of tasks to be executed
# tasks=("0" "1" "2" "3" "4")

# # Loop through the tasks and run the commands
# for task in "${tasks[@]}"; do
#     CUDA_VISIBLE_DEVICES=5 nnUNetv2_train 500 2d "$task"
# done


for fold in {0..4}
do 
    # echo "nnUNetv2_train 040 3d_lowres $fold
    nnUNetv2_train 040 3d_lowres $fold
done
