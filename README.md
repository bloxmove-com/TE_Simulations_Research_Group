# TE_Simulations
Token Engineering Research Group

## Integration of Intelligent Agents in cadCAD simulations

The aim of this research is to integrate, from the software archetecture point of view, intelligent agents in cadCAD simulations. The same workflow could be applied to TokenSPICE. 

#### Research questions

- intelligent agents are able to influence effectively the token ecosystem.
- intelligent agents are able to learn dynamically while interacting inside cadCAD simulator.

#### Ecosystem

We defined a test ecosystem where different actors, with different utility functions and economic power interact thorugh a decentralized exchanger. This DEX is in our case a Balancer pool with static weights equal to 0.5.

The agents we are considering are:
- investor: believes in the project and buy tokens regularly
- buyer: mainly buy tokens but in a smaller proportion and with doubts leading to some sells
- trader: actively participating in the pool both buying and selling

The logic of this agents is not optimized (not relevant for our research questions). 

Our agent:
- Foundation: has a interest in the sustainability of the pool and thorugh an intelligent and automatic agent, aims to influence optimizing several behaviors running from: fee values, dynamic wieghts on the pool, liquidity provision...

As a test of the integration of the intelligent agent in the platform, we selected a simple problem with an easy and understandable metric: token price. A lower value is predefined and the agent automatically keeps the value over the limit acting when something happens or is going to happen pushing the price down.

#### Intelligent Agents

When talking about behavior learning, reinforcement learning and deep reinforcement learning (RL) is the state-of-the-art approach. Gym.ai provides a set of libraries that ease the deployment, training and implementation of RL solutions. As a optimized oriented try and error solution, RL platform needs the control over the environment letting the agent explore the situation it wants. However, in an highly dynamic ecosystem where situations does not depend or not only depend on the actions of the RL agent, exploration and thus learning is limited. From the sw archetecture point of view: both gym and cadCAD are at the same hierarchical level difficulting the cooperation and integration of these two libraries. We show here the insights learnt during our participation in the research group:

- direct integration of both libraries is not possible
- cadCAD could be hierarchically downgraded and used inside the environment that gym requires for operating. This a complex solutions requiring developing time and programming effort.
- RL agents can be pretrained outside the cadCAD simulation and included afterwards. They are also able to learn during the simulation, however, this training process slows down cadCAD simulations and it is only recomended for a final fine tunning of the algorithm or dynamic adaptation. This is the easiest solution in terms of integration and does not penalize the final result. The definition of the agents cannot benefit form the gym library and the intelligent algorithms should be hardcoded what requires specific knowledege.

We have followed the last approach and include both the pretrained agent and the training loop in the cadCAD simulation. The utility function of our agent is simple but modifiying it for different use cases is also an easy process that only requires the description of the target as the labels used by the algorithm.

#### Results

Our results show the intelligent agent acting efficiently for the proposed task as can be seen in the different visualization options in the attached jupiter notebook and the independent training script (main_classification.py).

#### Conclusions

The integration of intelligent agents in cadCAD / tokenSPICE simulations is possible though not direct. Different options were analyzed and the easiest and most efficient way of integration seems to be the implementation of intelligent agents directly inside the cadCAD ecosystem. Specific knowledege is required in terms of machine learning algorithms and training flows. Training can be integrated in the cadCAD loop or externally and included in the loop as a pretrained model. We strongly recommend the second method due to the slow down in running time while training (simulation is slower and multiple simulation are necessary until the agent learns a meaningful behavior). A fine tunning of the intelligent agent, once pretrained, is possible and recommendable to dynamically adapt to new situations in the pool.

We are confident our work opens the door to furhter research in the combination of intelligent agents and token engineering in general and token ecosystem dynamics and DEX in particular.