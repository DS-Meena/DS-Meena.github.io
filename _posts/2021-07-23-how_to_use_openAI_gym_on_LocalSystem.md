# How to Use OpenAI Gym on your Local System

Setting up OpenAI gym is far easier than setting it on a server based notebook.

## Procedure

**Step 1: Install MS Visual C++ Build Tools for Visual Studio 2019**

This application is required to build the downloaded files, this is a must have software.
It will be used whenever you download a file that requires build tool.

This can be download [from here](https://visualstudio.microsoft.com/downloads/)

Click on > Tools for Visual Studio 2019
Then download Build Tools for Visual Studio 2019.
As seen in image

![Build tools for VS 2019](/images/build_tools_2019.png)

Tip - don't download all components, only choose those are required for building.

**Step 2: Install required Environments**

``pip install gym``

This will do a minimal installation of the OpenAI Gym.

1. ToyText environments:

    ``conda install pystan``

    This will install required packages(200MB) to run the ToxText environments. 
    This is optional, see whether you need this.

2. Atari environments:

    To simply install atari-py wheels(binaries) use this command:

    ``pip install -f https://github.com/Kojoley/atari-py/releases atari_py``

    Or if you have any distutils supported compiler you can install from sources:

    ``pip install git+https://github.com/Kojoley/atari-py.git``

3. Box2D environments:

    ``conda install swig``

    ``pip install Box2D``

    This will install packages required to run Box2D environments.

4. Remaining enviornments

    ``pip install gym[all]``

    This installs the remaining OpenAI Gym environments (ignore the errors for now).

To avoid bugs that can occur regarding Box2D and pyglet, run the following commands:

``pip install pyglet==1.2.4``

``pip install gym[box2d]``

**Step 3: Install Xming**

To see the environment running, you need to install Xming server on your computer.

It can be download from [here](https://sourceforge.net/projects/xming/) for free.

![Xming install](/images/install_xming.png)

**Step 4: Starting Xming server**

Whenver you want to use OpenAI Gym or to activate Xming server.
Run the following command to activate Xming server's window using command prompt.

``set DISPLAY=:0``

**Step 5: Test any Environment**

To test ny OpenAi Gym environment, run the following python code:

    import gym

    env = gym.make('env_name')
    env.reset()

    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())
    
    env.close()

If everything goes well, a new window will appear into your task bar menu.
Open it and you will be able to see the environment working.

Examles of different classes of Environments: -

1. Classic control

    Put env_name = "Acrobot-v1"

    ![OpenAI Gym environment](/images/OpenAI_Gym/Env_Acrobot-v1.png)

2. Box2D

    Put env_name = "BipedalWalker-v3"

    ![OpenAI Gym environment](/images/OpenAI_Gym/Env_BipedalWalker-v3.png)

3. Atari

    Put env_name = "Alien-v0"
    
    ![OpenAI Gym environment](/images/OpenAI_Gym/Env_Alien-v0.png)

4. Toy text 

    Put env_name = "FrozenLake-v0"

    ![OpenAI Gym environment](/images/OpenAI_Gym/Env_FrozenLake-v0.png)


That's it for now, i will add the remaining environments installtion process for you, when it will be avialble.

Be updated😉😉.