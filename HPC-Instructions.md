# HPC Instructions

## Discovery cluster information 
We will use CARC’s general-use HPC cluster "Discovery".

There are 2 ways to access this cluster: 
- SSH to the Discovery Cluster: You need to SSH from your terminal to the remote cluster. Please read the following [guide](https://www.carc.usc.edu/user-guides/hpc-systems/discovery/getting-started-discovery)
- OnDemand: This provides a graphical, browser-based access. Please read the following [guide](https://www.carc.usc.edu/user-guides/carc-ondemand/ondemand-overview)

## Connect to the USC network or Research VPN. [guide](https://www.carc.usc.edu/user-guides/hpc-systems/discovery/getting-started-discovery#connecting-to-the-usc-network-or-research-vpn)
You will have to either be connected to the `USC Secure Wireless` network or use the [USC VPN](https://www.carc.usc.edu/user-guides/quick-start-guides/anyconnect-vpn-setup) to connect to the cluster.

## Log in to Discovery node and access the terminal:
There are 2 ways:
- SSH to the Discovery Cluster: Refer to the [instructions](https://www.carc.usc.edu/user-guides/hpc-systems/discovery/getting-started-discovery#logging-in) based on your OS. 
- OnDemand: 
    1. Login using USC NetID.
    2. Click on Home Directory
    3. Open the terminal in the Home Directory. 
    
In both ways, you should get `[username@discovery* ~]` prompt in the terminal.

**Note: It is important to be familiar with how to use the shell for each case, and also how to transfer files.**

## Conventions:
In the rest of the document, we will assume the following convention:

- In code blocks, lines that start with `$` are shell commands. Copy the text after the `$`.
- In code blocks, lines that start with `#` are either outputs or comments.
- Output code in `<...>` are placeholders. In output lines that start with `#` just refers to text we don't necessarily care about.

## Initial Setup:

Before you can run the experiments, you will have to setup the environment for the HPC.

### Python/Anaconda setup

Run the following in the shell:

```shell
$ module purge
$ module load conda
$ conda init bash
$ conda config --set auto_activate_base false # This will prevent the base environment from being loaded
$ source ~/.bashrc
```

This will load the [Anaconda](https://docs.conda.io/en/latest/) module. So, you will have to create your own environment:

```shell
$ conda activate base
$ conda create -n carla python=3.8 conda -c conda-forge
$ ~/.conda/envs/carla/condabin/conda init bash # Singularity env is unable to load the conda installed on HPC. This changes it to the conda installed in the carla environment
```

This will create the `carla` environment with Python 3.8. To use this environment, run:

```shell
$ conda activate carla
```

To verify that you have the correct environment loaded, make sure the command line prompt begins with `(carla)` and the following commands output correctly:

```shell
$ python --version
# Python 3.8.<*>
$ which python
# ~/.conda/envs/carla/bin/python
```

### Code transfer

First, get the template for the project:

```shell
$ cd ~
$ cp /project/jdeshmuk_786/csci513-miniproject2.zip .
$ unzip csci513-miniproject2.zip
```

This will create a `csci513-miniproject2` directory in your home directory.
You can use the file transfer methods described in the linked guides above to update the controller in this directory with your code as needed.

### Acquire a compute node

Run the following command to acquire a compute "node" with a GPU:

```shell
$ salloc --time=2:00:00 --cpus-per-task=8 --mem=32GB --account=jdeshmuk_786 --partition=gpu --gres=gpu:1
# salloc: Granted job allocation <job id>
# salloc: Waiting for resource configuration
# salloc: Nodes <node-name> are ready for job
```

Run the command `nvidia-smi` to ensure the GPU drivers were loaded properly:

```shell
$ nvidia-smi
```

### Carla setup

```shell
$ module load conda
$ source ~/.bash_profile
$ conda activate carla
```

We will be using [`Singularity`](https://docs.sylabs.io/guides/3.7/user-guide/index.html) to run CARLA within a container.
Now, launch the singularity container:

```shell
$ singularity exec --nv /project/jdeshmuk_786/carla-0.9.15_4.11.sif bash
```

When you run this, your prompt should change to `Apptainer>`. Then, run the following to install the required Python library.

```shell
$ source ~/.bashrc
$ conda activate carla # The prompt should change to show that the carla env is activated
$ pip install -U carla==0.9.15
```

```shell
$ cd ~/csci513-miniproject2
$ python3 -m pip install -U -e .
```

Now, we will try to run Carla and verify that it works.
Run the following while still inside the Singularity container:

```shell
$ cd /home/carla/ # This will change directories to the specified one
$ bash ./CarlaUE4.sh -nosound -vulkan -RenderOffScreen & # The & is required
# chmod: changing permissions of '/home/carla/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping': Read-only file system
# 4.26.2-0+++UE4+Release-4.26 522 0
# Disabling core dumps.

# Hit ENTER here and the prompt will reappear. If it doesnt, make sure that you put in the &
$ cd PythonAPI/util
$ python3 test_connection.py
# CARLA 0.9.15 connected at 127.0.0.1:2000.
```

If you don't see the last output, then there is an issue.

**Note: To exit from the Singularity container and/or the allocated job node, simply run the command `exit`.**

You can exit the container and the job node now (you will have to run `exit` twice: once for the container, once for the job node).

## Running the experiment

First get a job allocation using `salloc`:

```shell
$ salloc --time=2:00:00 --cpus-per-task=8 --mem=32GB --account=jdeshmuk_786 --partition=gpu --gres=gpu:1
# salloc: Granted job allocation <job id>
# salloc: Waiting for resource configuration
# salloc: Nodes <node-name> are ready for job
```

Once you're allocated a node (based on the output above), you will run the following:

```shell
$ module load conda
$ source ~/.bash_profile
$ singularity exec --nv /project/jdeshmuk_786/carla-0.9.15_4.11.sif bash # You will get into the singularity container
$ source ~/.bashrc
$ conda activate carla # Your prompt should change to show that the env was activated
$ bash /home/carla/CarlaUE4.sh -nosound -vulkan -RenderOffScreen & # This will launch Carla. Remember to hit ENTER after seeing the desired output to get the prompt back.
$ cd ~/csci513-miniproject2
```

Now, you should be able to run your simulations and evaluate your designs as
described in the README file (follow the instructions after the Carla server has
been started).
