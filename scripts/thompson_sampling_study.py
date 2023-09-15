import re
import os
import pdb
import time
import torch
import pickle
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import beta

from verbal_gym.llm.gpt.gpt import GPT3
from verbal_gym.utils.tensorboard import Tensorboard


class BernoulliBandit:

    def __init__(self):
        self.num_actions = 2
        self.params = {
            0: 0.8,
            1: 0.5
        }

    def get_mean_return(self, action):
        return self.params[action]

    def step(self, action):

        assert action in [0, 1]

        r = random.random()
        if r < self.params[action]:
            return 1.0
        else:
            return 0.0


class ThompsonSampling:

    def __init__(self, num_actions, a=0.5, b=0.5, num_est=1000):
        self.num_actions = num_actions
        self.num_est = num_est

        self.params = []
        for _ in range(num_actions):
            self.params.append({"a": a, "b": b})

    def update_from_history(self, history):
        for action, reward in history:
            self.update(action, reward)

    def update(self, action, reward):
        self.params[action]["a"] = self.params[action]["a"] + reward
        self.params[action]["b"] = self.params[action]["b"] + 1 - reward

    def get_prob(self):

        all_action_samples = []
        for i in range(self.num_actions):
            action_samples = beta.rvs(a=self.params[i]["a"], b=self.params[i]["b"], size=self.num_est)
            all_action_samples.append(action_samples)

        action_prior = np.vstack(all_action_samples)  # K x num_est
        actions_chosen = np.argmax(action_prior, axis=0)  # num_est
        assert actions_chosen.shape[0] == self.num_est

        action_counts = np.zeros(self.num_actions).astype(np.float32)
        for action_chosen in actions_chosen:
            action_counts[action_chosen] += 1.0

        action_prob = action_counts / float(self.num_est)  # num_est

        return action_prob


class LLMAgent:

    def __init__(self, num_actions, use_log_prob=True, permute=True, num_permute=1, num_action_sample=5):

        self.num_actions = num_actions
        self.agent_history = []

        self.use_log_prob = use_log_prob
        self.permute = permute
        # No point using many permutations when not permuting
        self.num_permute = num_permute if self.permute else 1
        self.num_action_sample = num_action_sample

        self.llm = GPT3()

        self.base_prompt = "I am trying to solve a problem where I have two possible actions: action 1 and action 2. " \
                           "For each action, I get either a good reward or a bad reward. An action can give me both " \
                           "good or bad reward with different probabilities. I want to eventually take an action " \
                           "that gives the good reward with higher probability. In the past, I have taken the " \
                           "following actions and received the feedback as stated below:\n"

    def update(self, action, reward):
        self.agent_history.append((action, reward))

    def get_prob(self):

        batch_llm_probs = []
        for _ in range(self.num_permute):

            if self.permute:
                history_copy = list(self.agent_history)
                random.shuffle(history_copy)
            else:
                history_copy = self.agent_history

            if self.use_log_prob:
                prob = self.get_prob_via_logprob(history=history_copy)
            else:
                prob = self.get_prob_via_generation(history=history_copy)

            batch_llm_probs.append(prob)

        # Take mean
        batch_llm_probs = np.vstack(batch_llm_probs)        # num-permute x num-actions
        probs = np.mean(batch_llm_probs, axis=0)            # num-actions

        return probs

    def get_prob_via_generation(self, history):

        llm_probs = np.zeros(self.num_actions).astype(np.float32)
        for _ in range(self.num_action_sample):

            prompt = self.base_prompt
            prompt += "\n".join([f"- Took action {action} and got a good reward"
                                 if reward == 1 else f"- Took action {action} and got a bad reward"
                                 for action, reward in history])

            prompt += f"\n Based on the above feedback, I should choose action"

            logprob_action_string = self.llm.generate(prompt,
                                                      max_tokens=3,
                                                      timeout=300,
                                                      temperature=0.0,
                                                      max_attempts=1000)

            number_strings = re.findall(r'\d+', logprob_action_string)

            if len(number_strings) == 0:
                # Choose an action randomly
                action = random.randint(0, self.num_actions - 1)
                print(f"Warning! Didn't find any text in response {logprob_action_string}. \n Prompt was {prompt}.")
                pdb.set_trace()
            elif len(number_strings) == 1:
                action = int(number_strings[0])
            else:
                # Choose the first. The first number, when multiple are present, is a good hack as
                # it is closest to the prompt
                action = int(number_strings[0])
                print(f"Warning! Found more than one number in response "
                      f"{logprob_action_string}. \n Prompt was {prompt}.")
                pdb.set_trace()

            llm_probs[action] = llm_probs[action] + 1

        llm_probs = llm_probs / float(self.num_action_sample)

        return llm_probs

    def get_prob_via_logprob(self, history):

        logprob_actions = []
        for action in range(self.num_actions):
            prompt = self.base_prompt
            prompt += "\n".join([f"- Took action {action} and got a good reward"
                                 if reward == 1 else f"- Took action {action} and got a bad reward"
                                 for action, reward in history])

            prompt += f"\n Based on the above feedback, I should choose action {action}."

            logprob_action = self.llm.logprob(prompt)
            logprob_actions.append(logprob_action)

        logprobs = torch.FloatTensor(logprob_actions)
        llm_probs = torch.softmax(logprobs, dim=0)

        llm_probs = np.array([llm_probs[i].item() for i in range(self.num_actions)]).astype(np.float32)

        return llm_probs


class CalibrationStudy:

    def __init__(self, agent_type, save_path, num_eps=100, warmup_eps=5):

        self.agent_type = agent_type
        self.save_path = save_path
        self.num_eps = num_eps
        self.warmup_eps = warmup_eps

    @staticmethod
    def get_mean_returns_and_count(history, num_actions):

        sum_action = np.zeros(num_actions)
        counts = np.zeros(num_actions)

        for action, reward in history:
            sum_action[action] += reward
            counts[action] += 1

        mean_action_reward = sum_action / np.maximum(1, counts).astype(np.float32)
        return mean_action_reward, counts

    @staticmethod
    def take_action(prob):

        r = random.random()
        cumm = 0.0
        for i in range(len(prob)):
            if cumm <= r < cumm + prob[i]:
                return i
            cumm += prob[i]

        return len(prob) - 1

    def run(self, env, llm_agent, thompson_agent, tensorboard=None):

        results = []
        history = []

        for action in range(env.num_actions):
            for k in range(self.warmup_eps):
                reward = env.step(action)
                history.append((action, reward))

        for i in range(self.num_eps):

            # Parse the history and decide the next action
            print(f"Starting Round {i + 1}")
            mean_action_reward, counts = self.get_mean_returns_and_count(history, env.num_actions)

            llm_agent_prob = llm_agent.get_prob()
            thompson_agent_prob = thompson_agent.get_prob()

            mean_action_s = ", ".join([f"{mean_action_reward[i]:.3f}" for i in range(0, env.num_actions)])
            counts_s = ", ".join([f"{counts[i]}" for i in range(0, env.num_actions)])
            llm_agent_prob_s = ", ".join([f"{llm_agent_prob[i]:.3f}" for i in range(0, env.num_actions)])
            thompson_agent_prob_s = ", ".join([f"{thompson_agent_prob[i]:.3f}" for i in range(0, env.num_actions)])

            print(f"Episode {i + 1}: Action Return {mean_action_s} and Action Counts {counts_s}.")
            print(f"Episode {i + 1}: LLM Agent Prob {llm_agent_prob_s}")
            print(f"Episode {i + 1}: Thompson Sampling Agent Prob {thompson_agent_prob_s}")

            if self.agent_type == "llm":
                prob = llm_agent_prob
            elif self.agent_type == "thompson":
                prob = thompson_agent_prob
            else:
                raise AssertionError(f"Unhandled agent type {self.agent_type}")

            action = self.take_action(prob)
            reward = env.step(action)
            history.append((action, reward))

            print(f"Episode {i + 1}: Took action {action} and got reward {reward} using agent type {self.agent_type}")

            # Update the agents.
            thompson_agent.update(action, reward)
            llm_agent.update(action, reward)

            result = {
                "mean_action_return": mean_action_reward,
                "action_counts": counts,
                "llm_prob": llm_agent_prob,
                "thompson_prob": thompson_agent_prob,
                "action": action,
                "reward": reward,
            }
            results.append(result)

            if tensorboard is not None:
                tensorboard.log_scalar("LLM Agent action 0 prob", llm_agent_prob[0])
                tensorboard.log_scalar("Thompson Sampling action 0 prob", thompson_agent_prob[0])
                tensorboard.log_scalar("action", action)
                tensorboard.log_scalar("reward", reward)

                tensorboard.log_scalar("mean action 0 return", mean_action_reward[0])
                tensorboard.log_scalar("gold action 0 return", env.get_mean_return(0))

                tensorboard.log_scalar("mean action 1 return", mean_action_reward[1])
                tensorboard.log_scalar("gold action 1 return", env.get_mean_return(1))

            print("\n\n")

        print("Experiment Over. Generating plot and saving the data.")

        return results

    def plot(self, results):

        self.plot_action_prob(results, fname="probabilities.png")

        self.plot_action_return(results, fname="action_return.png")

    def plot_action_prob(self, results, fname):

        plt.clf()

        # Plot returns
        plt.title(f"Bernoulli Bandit with 2 actions. Agent: {self.agent_type}, Warmup: {self.warmup_eps}.")
        plt.ylabel("Action 0 probability")
        plt.xlabel("Episodes")
        episodes = list(range(1, self.num_eps + 1))

        # Plot probabilities of action-1 for each LLM (since there is only 2 action in the study,
        # the other is automatically determined)
        plt.plot(episodes, [result["llm_prob"][0] for result in results],
                 label="LLM Prob. of action 0", color="blue")

        plt.plot(episodes, [result["thompson_prob"][0] for result in results],
                 label="Thompson Prob. of action 0", color="green")

        plt.legend()
        plt.savefig(f"{self.save_path}/{fname}")

    def plot_action_return(self, results, fname):

        plt.clf()
        plt.title("Mean action return vs episodes")
        plt.ylabel("Mean return")
        plt.xlabel("Episodes")

        episodes = list(range(1, self.num_eps + 1))

        plt.plot(episodes, [env.get_mean_return(0)] * self.num_eps,
                 label="Mean return of action 0", color="blue", linestyle="--")
        plt.plot(episodes, [result["mean_action_return"][0] for result in results],
                 label="Emp. mean return of action 0", color="blue")

        action_1_eps = [i + 1 for i, result in enumerate(results) if result["action"] == 0]
        action_1_indicator = [result["mean_action_return"][0] for result in results if result["action"] == 0]

        plt.plot(action_1_eps, action_1_indicator, label="_nolegend_", ls="", color="blue", marker="o")

        plt.plot(episodes, [env.get_mean_return(1)] * self.num_eps,
                 label="Mean return of action 1", color="green", linestyle="--")
        plt.plot(episodes, [result["mean_action_return"][1] for result in results],
                 label="Emp. mean return of action 1", color="green")

        action_2_eps = [i + 1 for i, result in enumerate(results) if result["action"] == 1]
        action_2_indicator = [result["mean_action_return"][0] for result in results if result["action"] == 1]

        plt.plot(action_2_eps, action_2_indicator, label="_nolegend_", ls="", color="green", marker="o")

        plt.legend()
        plt.savefig(f"{self.save_path}/{fname}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Study hyperparam
    parser.add_argument("--name", default="llm-calibration-study", type=str, help="Name of the experiment")
    parser.add_argument("--save_dir", default="./llm-calibration-studies", help="Save directory")
    parser.add_argument("--num_eps", default=100, type=int, help="number of episodes")
    parser.add_argument("--warmup_eps", default=0, type=int, help="number of warmup episodes")
    parser.add_argument("--agent", default="thompson", type=str, help="which agent to use for decision making",
                        choices=["llm", "thompson"])

    # Thompson sampling hyperparam
    parser.add_argument("--a", default=0.5, type=float, help="alpha hyperparam for Thompson prior")
    parser.add_argument("--b", default=0.5, type=float, help="beta hyperparam for LLM prior")
    parser.add_argument("--thompson_num_est", default=1000, type=int, help="number of samples for Thompson sampling")

    # LLM sampling hyperparam
    parser.add_argument("--use_log_prob", default=1, type=int, help="if > 0 then use log_prob for LLM agent "
                                                                      "else dont use")
    parser.add_argument("--permute", default=1, type=int, help="if > 0 then permute history else dont")
    parser.add_argument("--num_permute", default=1, type=int, help="if > 0 then permute history else dont")
    parser.add_argument("--num_action_sample", default=5, type=int, help="if LLM is in generation mode, then we can "
                                                                         "sample actons many time to generate an "
                                                                         "empirical distribution from which we actually"
                                                                         " take the action")

    args = parser.parse_args()
    exp_name = f"{args.name}-{int(time.time())}"
    save_path = f"{args.save_dir}/{exp_name}"

    if os.path.exists(save_path):
        raise AssertionError(f"Save Path {save_path} already exists")
    else:
        os.makedirs(save_path)

    env = BernoulliBandit()

    llm_agent = LLMAgent(num_actions=env.num_actions,
                         use_log_prob=args.use_log_prob > 0,
                         permute=args.permute > 0,
                         num_permute=args.num_permute,
                         num_action_sample=args.num_action_sample)

    thompson_agent = ThompsonSampling(num_actions=env.num_actions,
                                      a=args.a,
                                      b=args.b,
                                      num_est=args.thompson_num_est)

    study = CalibrationStudy(agent_type=args.agent,
                             save_path=save_path,
                             num_eps=args.num_eps,
                             warmup_eps=args.warmup_eps)

    if not os.path.exists(f"{save_path}/tensorboard"):
        os.makedirs(f"{save_path}/tensorboard")

    tensorboard = Tensorboard(log_dir=f"{save_path}/tensorboard")

    results = study.run(env=env,
                        llm_agent=llm_agent,
                        thompson_agent=thompson_agent,
                        tensorboard=tensorboard)

    # Save plots
    study.plot(results)

    # Save setting
    setting = {}

    for k, v in vars(args).items():
        setting[k] = v

    data = {
        "setting": setting,
        "results": results
    }

    with open(f"{save_path}/results.pkl", "wb") as f:
        pickle.dump(data, f)