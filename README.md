# counterfactual

##installation_instructions
conda create -n causal_rl_bench python==3.7.4 

conda activate causal_rl_bench

conda config --add channels conda-forge

conda install pinocchio==2.2.2

cd causal_rl_bench

pip install -r requirments.txt

pip install -e .

cd pybullet_fingers

pip install -e .


