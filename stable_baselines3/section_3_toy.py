import numpy as np
import torch

initial_theta = torch.tensor(np.random.uniform(low=-1, high=1, size=(1, 1)), dtype=torch.float32)
initial_omega = torch.tensor(np.random.uniform(low=-1, high=1, size=(1, 1)), dtype=torch.float32)
def reward_function(theta):
    return -(1/5)*theta**2

raw_input = np.random.normal(loc=1, scale=1, size=(100, 1))
theta = np.tanh(raw_input) # constrain the input to be between -1 and 1
reward = reward_function(theta)

critic_model = torch.nn.Sequential(
    torch.nn.Linear(1, 1, bias=False),
)
actor_model = torch.nn.Sequential(
    torch.nn.Linear(1, 1, bias=False),
)
critic_optimizer = torch.optim.SGD(critic_model.parameters(), lr=0.0001)
actor_optimizer = torch.optim.SGD(actor_model.parameters(), lr=0.001)
batch_size = 2
for i in range(5000):
    rewards = []
    actions = []
    states = []
    critic_preds = []
    # sample using actor
    for i in range(100):

        state = torch.tensor(np.random.uniform(low=-1, high=1, size=(100, 1)), dtype=torch.float32)
        actor_actions = actor_model(state)
        critic_pred = critic_model(actor_actions)
        reward = reward_function(actor_actions)
        rewards.append(reward)
        actions.append(actor_actions.detach())
        states.append(state.detach())
    # update actor
    for i in range(10):
        state = torch.cat(states)
        action = actor_model(state)
        output = critic_model(action)
        critic_loss = torch.mean((torch.cat(rewards) - output)**2)
        actor_loss = -torch.mean(output)
        critic_optimizer.zero_grad()
        grad = torch.autograd.grad(critic_loss, critic_model.parameters(), create_graph=True, retain_graph=True)
        critic_optimizer.zero_grad()
        for i in range(len(grad)):
            critic_optimizer.param_groups[0]['params'][i].grad = grad[i]
        
        actor_optimizer.zero_grad()
        #grad = torch.autograd.grad(actor_loss, actor_model.parameters())
        for param in critic_model.parameters():
            pass
        for i in range(len(grad)):
            actor_optimizer.param_groups[0]['params'][i].grad = param.data.detach()
        critic_optimizer.step()
        actor_optimizer.step()
    #print(actor_loss.item(), critic_loss.item())
    for param in critic_model.parameters():
        print(param.data)
    print("--------------------------------")
    for param in actor_model.parameters():
        print(param.data)