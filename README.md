# mm_sequential_tasks
## Run Isaac Sim Simulator Only
1. Make sure [mmseq_sim_isaac](https://github.com/TracyDuX/mmseq_sim_isaac) has been installed.
2. Run
   ```
   roslaunch mmseq_run isaac_sim.launch config:=$(rospack find mmseq_run)/config/3d_collision.yaml isaac-venv:=$ISAACSIM_PYTHON
   ```
   where `$ISAACSIM_PYTHON' is the `./python.sh` file in the Isaac Sim root folder.
