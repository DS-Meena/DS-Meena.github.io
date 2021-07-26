# How to use OPEN AI gym on server

In this disussion, we will try to learn how we can run the [OpenAI gym](https://gym.openai.com/envs/#classic_control) environment on Jupyter notebook server platforms like colab and Paperspace gradient notebook.

## OpenAI Gym

It provides a standard benchmark with a wide variety of different environments. You can read in more detail from [here](https://github.com/jeffheaton/t81_558_deep_learning/blob/master/t81_558_class_12_01_ai_gym.ipynb).  

## Setup to run gym  

Install the required libraries for running the openAI gym environments.  

- install the basic library

``!pip install gym``

- install librarires required to see the environment working in a display in your notebook (running in a server):

``!pip install gym pyvirtualdisplay``

``!apt-get install -y xvfb python-opengl ffmpeg``

- and run these commands.  

``!apt-get update``

``!apt-get install cmake``

``!pip install --upgrade setuptools``

``!pip install ez_setup``

``!pip install gym[atari]``

## Try OpenAI Gym

## Get to know environment

Define function, that will print info about the given environment id.  

    import gym

    def query_environment(name):
        env = gym.make(name)
        spec = gym.spec(name)
        print(f"Action Space: {env.action_space}")
        print(f"Observation Space: {env.observation_space}")
        print(f"Max Episode Steps: {spec.max_episode_steps}")
        print(f"Nondeterministic: {spec.nondeterministic}")
        print(f"Reward Range: {env.reward_range}")
        print(f"Reward Threshold: {spec.reward_threshold}")

To know about any OpenAI Gym environment, use the following code

    query_environment("env_id")

Example -

    query_environment("CartPole-v1")

    # output
    Action Space: Discrete(2)
    Observation Space: Box(-3.4028234663852886e+38, 3.4028234663852886e+38, (4,), float32)
    Max Episode Steps: 500
    Nondeterministic: False
    Reward Range: (-inf, inf)
    Reward Threshold: 475.0

From this, result you can infer many things about environment like no of actions possible, observation space size, max episode length, whether it is non deterministic or deterministic, reward range and reward threshold.

## See how it is running

Define function to wrap environment and to save the environment run as a video.

    import gym
    import random
    from gym.wrappers import Monitor
    import glob
    import io
    import base64
    from IPython.display import HTML
    from pyvirtualdisplay import Display
    from IPython import display as ipythondisplay

    display = Display(visible=0, size=(1400, 900))
    display.start()

    """
    Utility functions to enable video recording of gym environment 
    and displaying it.
    """

    def show_video():
        mp4list = glob.glob('video/*.mp4')
        if len(mp4list) > 0:
            mp4 = mp4list[0]
            video = io.open(mp4, 'r+b').read()
            encoded = base64.b64encode(video)
            ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
        else: 
            print("Could not find video")
    

    def wrap_env(env):
        env = Monitor(env, './video', force=True)
        return env

Define your agent, here i have initialized a random agent, that will take random actions in the environment.

    class Agent():
    def __init__(self, env):
        print("initialized")
    
    def get_action(self, state):
        action = env.action_space.sample()  # return a random action
        return action

After doing the above defining functions part, now you can do the real thing.
Create an environment and run it using your agent.
It will stop when the episode terminates, then it will display the saved environment video.

**Note - To enable video, just do "env = wrap_env(env)"** 

    env = wrap_env(gym.make("MountainCar-v0")) # enable video
    state = env.reset()

    # create an agent
    agent = Agent(env)

    while True:
        # get action from the agent
        action = agent.get_action(state)  
    
        # observe new state
        state, reward, done, info = env.step(action) 
   
        env.render()
    
        if done:   # end episode 
            break;
            
    env.close()
    show_video()

Output

![copy api command](/images/OpenAI_Gym/mountaincar-v0.png)

## Run Atari and complex environment (ROM Issue)

Some envrionments require rom to run them. 
You can try with the above code, but it will show missing rom error in your server notebook.

Example:  

    Exception: ROM is missing for pong, see https://github.com/openai/atari-py#roms for instructions

So, we need to download the rom, and import it during running our environments.

1. Download the Rar file from the website  http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html. (you need to download a 10MB ROMS.rar file from here).

2. Unrar the ROMS.rar file, using any rar extracting software and extract both zip files from it. (you can try code, but why making it difficult)

3. Upload both zip files (HC ROMS & ROMS) into your server notebook like colab or gradient, manually.

4. Create a folder (rars), move both zip files into it.

    ``!mkdir rars``

    ``!mv HC\ ROMS.zip   rars``  
    ``!mv ROMS.zip  rars``

5. Now import the rom using below code.

    ``!python -m atari_py.import_roms rars``

6. Now you can run complex environments.  

Example :- try the pong game, just change the env id in above code, and run it after importing rom.  

    env = wrap_env(gym.make("PongNoFrameskip-v4"))

Output: -

![copy api command](/images/OpenAI_Gym/pongNoFrameSkip-v4.png)

## Summary

- For simple environments, Just define show_video() and wrap_env() functions in any notebook. 
- Then Enable video, using "env = wrap_env(env)" & then you will be able to see it working.
- For complex environments (that require ROM), just move the rars folder to local directory
- import ROM using ``!python -m atari_py.import_roms rars`` and now you will be able to work with complex environments.

That's was enough for this small blog.