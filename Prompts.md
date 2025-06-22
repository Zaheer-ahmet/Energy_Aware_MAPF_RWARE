*** Desired_algorithm.py Creation Prompt ***

okay so moving on, I would like you to write a similar script to evaluate, in this script named: "Desired_algorithm";
-There will be a layout similar to the one in evaluate but with addition of charging stations.
-The logic of rewards, both individual and team, will be the same as "evaluate.py".
-There will be two agents & two goals, 1 charging stations.
-The initial script will be a human play to test the rewards and charging station logic and it will be episodic as is in @human_play_hrpf.py .
-In each episode, the individual reward is reset but the team reward is cumulative.
-There will be no max steps, termination will only be initiated when esc button is pressed as is in "human_play_hrpf"
-Much like "human_play_hrpf", there will be a list that shows the individual reward counter for each agent with the addition of team rewards to the left of it.

During creating this new script DO NOT CHANGE hrpf_team_warehouse.py, instead I have copied the warehouse as "Comb_warehouse.py", you can edit that warehouse if necessary and remove the RL components which was necessary for training and is redundant now.

*** Charging_Station addition to Human_Play ***
I want you to update the log with the last three prompts we did, Improvements done to RL and bug fix, and lastly the human play we created.
After that, edit the "Desired_algorithm.py" file such that report for each step is not printed in the terminal, only the last state of rewards. instead these step by step report are saved under /Users/zaheer_ahmad/rware-reward-charge/Desired_algorithm/Human_Play.
Also you forgot to add the charging stations which were represented by green tiles and the agents restored their battery when passing there, the logic of it was present inside @hrpf_team_warehouse.py , where each movement step caused the agent to deplete 1 from its battery life and when passing the stations it restored to the max capacity.
add that also to the layout and logic and update the log again in the end.

*** Reward structure ***
I think the issue emerges when an agent has reached its goal as a last step of that episode and its charge becomes 0 and the goal position is set as a ccharging station, so when the episode is reloaded the agents seem to think that that charging station is the default goal. Also the rewards, penalty and episode counters seem to malfuction.
-Episode counter should increase by 1 when a new episode is initiated.
-Each agent should receive a penalty of -0.1 when performing a successfull or failed forward move, the total team reward should take a penalty of -0.1 with each step passed.
-Each agent should receive a reward of 10 for raching its goal (only the default goal, the charging stations as goals do not count)
-Team reward should be increased by 50 when both of the agents have reached their goals, in other words when an episode is finished successfully.
-Individual rewards should reset at the start of each episode, the team rewards and battery life should not reset
Check the script for the above rules and examine whether they are implemented correctly, debug if not.

*** Sensitivity Test Log Generator ***
okay perfect, now I would like you to create a separate output file for each run of the simulation now that all debugging is finished and the testing phase begins. Each run should yield an ouptut file in the following format:
"
Run ID: <INSERT_RUN_ID_HERE>

Episode 1:
Steps taken: [<int>, <int>, <int>, <int>]
Recharges: [<int>, <int>, <int>, <int>]
Episode Duration: <int>
Individual Rewards: [<float>, <float>, <float>, <float>]
Team Reward: <float>

Episode 2:
Steps taken: [<int>, <int>, <int>, <int>]
Recharges: [<int>, <int>, <int>, <int>]
Episode Duration: <int>
Individual Rewards: [<float>, <float>, <float>, <float>]
Team Reward: <float>

...

Total Accumulated Team Reward: <float>
Total Steps Taken: <int>
Total Number of Episodes: <int>
"
Here is addtional notes explaining these values:
"Steps taken = steps for each agent before completing its goal
Recharges = number of recharges per agent for BFS_Charge (put [0,0,0,...] for No_Charge)
Episode Duration = total time (step count) for all agents to finish
Individual Rewards = final episodic reward each agent receives (could include penalties + goal reward)
Team Reward = final team reward given for that episode
"
Check if the script already has the necessary parameters and variables set to obtain these values as report. if not add them and add this format into both of the scripts:"Desired_algorithm_BFS_Charge.py" and "Desired_algorithm_BFS_No_Charge.py" with the exception of "#  ofRecharges" in the latter script because it does not apply.
Let me note that the run ID will be provided in the trigger code beside the max episode and max battery.
