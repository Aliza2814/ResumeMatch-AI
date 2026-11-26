"""
Reinforcement Learning Experiment for ResumeMatch AI (Unit 10).
Demonstrates an agent interacting with an environment through Markov Decision Process (MDP):
- Agent: Job Candidate / Career Optimizing Agent
- Environment: Job Market & Application Simulator
- State: (Current Skill Level [Low, Med, High], Resume Tailoring Quality [Low, High], Experience [Junior, Senior])
- Actions: [0: Upskill/Study, 1: Tailor Resume, 2: Apply to General Jobs, 3: Apply to Targeted Jobs]
- Rewards:
    + Interview invite: +10
    + Job offer: +50
    + Skill upgrade: +5
    + Rejection / Application cost: -2
- Algorithm: Tabular Q-Learning with epsilon-greedy exploration.
"""

import sys
from pathlib import Path
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

class JobSearchEnvironment:
    """
    Simulated Career & Job Application MDP Environment.
    State space: 3 discrete factors -> 3 x 2 x 2 = 12 total discrete states.
    Actions: 4 actions [Upskill, Tailor, Apply General, Apply Targeted].
    """
    def __init__(self):
        # States: (skill_level: 0, 1, 2), (tailored: 0, 1), (exp_level: 0, 1)
        self.num_states = 3 * 2 * 2
        self.num_actions = 4
        self.action_names = {
            0: "Upskill (Acquire Tech Skill)",
            1: "Tailor Resume for Specific Role",
            2: "Apply to General Jobs",
            3: "Apply to Targeted High-Match Jobs"
        }
        self.reset()

    def reset(self):
        self.skill_level = 0  # 0: Low, 1: Medium, 2: High
        self.tailored = 0     # 0: Generic, 1: Tailored
        self.exp_level = 0    # 0: Junior, 1: Experienced
        self.steps = 0
        return self._get_state_index()

    def _get_state_index(self):
        return self.skill_level * 4 + self.tailored * 2 + self.exp_level

    def step(self, action: int):
        self.steps += 1
        reward = 0
        done = False

        if action == 0:  # Upskill
            if self.skill_level < 2:
                self.skill_level += 1
                reward = 5  # Positive investment
            else:
                reward = 1  # Already high skills
        elif action == 1:  # Tailor Resume
            self.tailored = 1
            reward = 3  # Tailoring effort
        elif action == 2:  # Apply to General Jobs
            # Probability of success depends heavily on skills and tailoring
            success_prob = 0.10 + 0.15 * self.skill_level + 0.15 * self.tailored
            if np.random.rand() < success_prob:
                reward = 25  # Offer received
                done = True
            else:
                reward = -2  # Rejection penalty / time loss
        elif action == 3:  # Apply to Targeted High-Match Jobs
            # Higher success if both skills are high and resume is tailored
            if self.tailored == 1 and self.skill_level >= 1:
                success_prob = 0.55 + 0.20 * self.skill_level
            else:
                success_prob = 0.15
            
            if np.random.rand() < success_prob:
                reward = 50  # Top Offer!
                done = True
            else:
                reward = -3  # Application friction

        if self.steps >= 20:  # Max horizon per episode
            done = True

        return self._get_state_index(), reward, done, {}

class QLearningAgent:
    """
    Tabular Q-Learning Agent.
    Updates Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
    """
    def __init__(self, num_states: int, num_actions: int, alpha: float = 0.1, gamma: float = 0.95, epsilon: float = 0.2):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((num_states, num_actions))

    def choose_action(self, state: int) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        return int(np.argmax(self.q_table[state]))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + (0 if done else self.gamma * self.q_table[next_state, best_next_action])
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

def run_reinforcement_learning():
    print("=== Running Unit 10: Reinforcement Learning & Emerging Trends ===")
    env = JobSearchEnvironment()
    agent = QLearningAgent(num_states=env.num_states, num_actions=env.num_actions, alpha=0.15, gamma=0.9, epsilon=0.3)

    episodes = 500
    rewards_history = []
    rolling_avg = []

    print(f"  Training Q-Learning Agent across {episodes} episodes...")
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        # Decay exploration epsilon
        if agent.epsilon > 0.05:
            agent.epsilon *= 0.995

        rewards_history.append(total_reward)
        if len(rewards_history) >= 20:
            rolling_avg.append(np.mean(rewards_history[-20:]))
        else:
            rolling_avg.append(np.mean(rewards_history))

    # Save Q-table
    joblib.dump({"q_table": agent.q_table, "action_names": env.action_names}, MODELS_DIR / "rl_q_learning.joblib")

    # Plot learning curve
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history, color="#94A3B8", alpha=0.4, label="Episode Cumulative Reward")
    plt.plot(rolling_avg, color="#2563EB", linewidth=2.5, label="20-Episode Moving Average")
    plt.title("Unit 10: Reinforcement Learning - Candidate Agent Learning Curve", fontsize=13, fontweight="bold")
    plt.xlabel("Episode (Application Cycle)", fontsize=11)
    plt.ylabel("Cumulative Reward (Offers & Skills - Cost)", fontsize=11)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "rl_learning_curve.png", dpi=150)
    plt.close()

    # Document Optimal Policy learned by the Agent
    optimal_policy = {}
    for s in range(env.num_states):
        best_a = int(np.argmax(agent.q_table[s]))
        skill = s // 4
        tailored = (s % 4) // 2
        exp = s % 2
        state_desc = f"Skill={['Low', 'Med', 'High'][skill]}, Tailored={bool(tailored)}, Exp={['Junior', 'Senior'][exp]}"
        optimal_policy[state_desc] = env.action_names[best_a]

    results = {
        "episodes_trained": episodes,
        "final_20_ep_reward_avg": round(float(rolling_avg[-1]), 2),
        "initial_20_ep_reward_avg": round(float(rolling_avg[19]), 2),
        "optimal_policy": optimal_policy,
        "theoretical_rationale": "While RL effectively models multi-step career trajectory planning, content-based matching with TF-IDF/Cosine similarity and supervised classification are deterministic, explainable, and instantaneous for resume screening."
    }

    with open(REPORTS_DIR / "rl_metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"  [OK] RL experiment complete. Final Avg Reward: {rolling_avg[-1]:.2f}")
    return results

if __name__ == "__main__":
    run_reinforcement_learning()
