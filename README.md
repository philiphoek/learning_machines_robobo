# ROS ROBOBO FOR LEARNING MACHINES

Note, if you feel like not using this path but installing directly on your system, you are free to do it, but you are on your own.

## Download/Install requirements
Here are the instructions to have the simulation system running.

- Download VREP from <a href="http://www.coppeliarobotics.com/downloads.html">here</a>, V-REP PRO EDU V3.5.0 rev4
  - extract folder anywhere, where you preferer

- Download `Robobo_Scene.ttt` and `Robobo_Model.ttm` files and place them inside your VREP folder
  - place Robobo_Scene.ttt file in VREP/scenes
  - place Robobo_Scene.ttm file in VREP/models/robots/mobile

- Download docker
  - Linux: download from your package manager
  - OsX: register and download from [here](https://hub.docker.com/editions/community/docker-ce-desktop-mac)
  - Windows PRO: register and download from [here](https://hub.docker.com/editions/community/docker-ce-desktop-windows)
  - Windows HOME: register and download from [here](https://docs.docker.com/toolbox/toolbox_install_windows/)

- Download git + Editor

- Download repository
  - <pre>git clone <a href="https://github.com/portaloffreedom/learning_machines_robobo.git">https://github.com/portaloffreedom/learning_machines_robobo.git</a></pre>

## Running
Now you downloaded everything, you can start the environment

- Start docker&nbsp;
  - Linux/systemd: start docker daemon
  - OsX/Windows: open docker app

- Start docker container (this will open a terminal with python and all required libraries installed). This step could take a while, since first your computer will download the docker image from the internet, secondly will compile some stuff from your project folder.
  - Linux/OsX, open a terminal in the <strong><span style="color: #000000;">learning_machines_robobo</span></strong><span style="color: #000000;"> project</span>&nbsp;folder and type:
  
    ```./start-docker.sh```
  - Windows, open CMD and change directory to the&nbsp;<strong>learning_machines_robobo</strong> project folder, and type:
  
    ```start-docker.bat```
- Start VREP
  - open the program
  - open the scene file that you downloaded before: Robobo_Scene.ttt
  - enable Real Time button (optional)
  - start the simulation with the play button
  
- Fix the script: inside the project folder, open the file `src/send_commands.py` and
  - remove the line in which I create a connection to an hardware robot
  - fix the line of the connection to the simulated robot to have your local machine ip (localhost or 127.0.0.1 will not work because of docker. Connect to your home network and use the IP that the router will give you)

- Run the script
  - inside the docker container running, type
  ```./src/send_commands.py```

- Change the script, experiment and HAVE FUN :D

# Informations to run this project alternatevely

## Setup (with ros image)

- start docker with `docker run --rm -it -v "${PROJECT_FOLDER}:/root/projects/" cigroup/learning-machines bash`
- change folder to `/root/projects/` (inside docker)
- run `catkin_make install` (only run for the first setup, it's not needed even across different docker containers)
- run `source devel/setup.bash`
- run `src/send_commands.py` command to test the connection

### Important Notes

The environment variable `ROS_MASTER_URI` idendifies where the ROS MASTER NODE is running. Verify that your phone is in "Local ROS Master" and adjust the IP of the `ROS_MASTER_URI` variable with the address of the phone (visible at the first page of the application).

The library we provide is already taking care of this enviroment variable, but in case you encouter any problems you know where to look for.
