from __future__ import annotations

from pathlib import Path

import matplotlib

# 使用非交互式后端，避免服务器或本机Tk配置异常时无法保存图片。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import torch

from .dqn import DQNAgent
from .env import BeerGameEnv
from .policies import build_policy
from .ppo import PPOAgent
from .sac_discrete import DiscreteSACAgent
from .vec_env import VectorizedBeerGameEnv


def setup_chinese_font():
    """为Matplotlib注册中文字体，避免保存图片时中文显示为方块。

    优先尝试系统常见中文字体，找不到则使用 Matplotlib 默认无衬线字体
    并关闭 unicode_minus 以正常显示负号。
    """

    candidates = [
        # Windows
        Path("C:/Windows/Fonts/NotoSansSC-VF.ttf"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
        # Linux 常见中文字体（AR PL UMing 同时覆盖中文与拉丁字符，优先使用）
        Path("/usr/share/fonts/truetype/arphic/uming.ttc"),
        Path("/usr/share/fonts/truetype/arphic-gbsn00lp/gbsn00lp.ttf"),
        Path("/usr/share/fonts/truetype/arphic-gkai00mp/gkai00mp.ttf"),
        Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for font_path in candidates:
        if font_path.exists():
            try:
                font_manager.fontManager.addfont(str(font_path))
                font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
                plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
                plt.rcParams["axes.unicode_minus"] = False
                return
            except Exception:
                continue
    # 兜底：使用默认 sans-serif，至少保证程序不报错
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


setup_chinese_font()


def make_background_actions(
    env: BeerGameEnv,
    state: np.ndarray,
    firm_id: int,
    rng: np.random.Generator,
    background_policy: str = "random",
    target_inventory: int | None = None,
):
    if background_policy == "random":
        # 未被控制的企业视为环境的一部分，默认沿用随机订货背景。
        actions = rng.integers(0, env.config.max_order + 1, size=env.num_firms).astype(np.float32)
    elif background_policy == "base_stock":
        # 其他企业按库存补足规则订货，使需求能沿链路更自然地传导。
        target = env.config.initial_inventory if target_inventory is None else target_inventory
        actions = np.zeros(env.num_firms, dtype=np.float32)
        for i in range(env.num_firms):
            inventory = float(state[i, 2])
            actions[i] = np.clip(round(target - inventory), 0, env.config.max_order)
    else:
        raise ValueError(f"未知背景策略: {background_policy}")
    return actions


def make_background_actions_vec(
    env_vec: VectorizedBeerGameEnv,
    states: np.ndarray,
    firm_id: int,
    rng: np.random.Generator,
    background_policy: str = "random",
    target_inventory: int | None = None,
) -> np.ndarray:
    """Generate background actions for all parallel environments."""
    n, m = env_vec.num_envs, env_vec.num_firms
    max_order = env_vec.config.max_order
    actions = np.zeros((n, m), dtype=np.float32)
    if background_policy == "random":
        actions = rng.integers(0, max_order + 1, size=(n, m)).astype(np.float32)
    elif background_policy == "base_stock":
        target = env_vec.config.initial_inventory if target_inventory is None else target_inventory
        inventory = states[:, :, 2]
        actions = np.clip(np.rint(target - inventory), 0, max_order).astype(np.float32)
    else:
        raise ValueError(f"未知背景策略: {background_policy}")
    return actions


def evaluate_policy(
    env: BeerGameEnv,
    policy,
    firm_id: int,
    episodes: int,
    seed: int | None = 123,
    background_policy: str = "random",
    background_target_inventory: int | None = None,
):
    rng = np.random.default_rng(seed)
    scores = []
    histories = {"orders": [], "inventory": [], "demand": [], "satisfied": [], "rewards": []}
    for episode in range(episodes):
        state = env.reset(seed=None if seed is None else seed + episode)
        if hasattr(policy, "reset"):
            policy.reset()
        done = False
        score = 0.0
        ep = {key: [] for key in histories}
        while not done:
            actions = make_background_actions(
                env,
                state,
                firm_id,
                rng,
                background_policy=background_policy,
                target_inventory=background_target_inventory,
            )
            # 只把目标企业的随机动作替换为待评估策略的动作。
            actions[firm_id] = policy.act(state, firm_id)
            next_state, rewards, done, info = env.step(actions)
            score += float(rewards[firm_id, 0])
            ep["orders"].append(float(info["actions"][firm_id]))
            ep["inventory"].append(float(info["inventory"][firm_id]))
            ep["demand"].append(float(info["demand"][firm_id]))
            ep["satisfied"].append(float(info["satisfied_demand"][firm_id]))
            ep["rewards"].append(float(rewards[firm_id, 0]))
            state = next_state
        scores.append(score)
        for key in histories:
            histories[key].append(ep[key])
    return {"scores": np.asarray(scores, dtype=np.float32), "histories": histories}


def train_dqn(env: BeerGameEnv, agent: DQNAgent, cfg: dict):
    rng = np.random.default_rng(cfg.get("seed", 42))
    background_policy = str(cfg.get("background_policy", "random"))
    background_target = int(cfg.get("background_base_stock_target", env.config.initial_inventory))
    scores = []
    eps = float(cfg.get("eps_start", 1.0))
    eps_end = float(cfg.get("eps_end", 0.01))
    eps_decay = float(cfg.get("eps_decay", 0.995))
    episodes = int(cfg.get("episodes", 500))

    for episode in range(1, episodes + 1):
        state = env.reset(seed=int(cfg.get("seed", 42)) + episode)
        done = False
        score = 0.0
        while not done:
            actions = make_background_actions(
                env,
                state,
                agent.firm_id,
                rng,
                background_policy=background_policy,
                target_inventory=background_target,
            )
            firm_state = state[agent.firm_id]
            # DQN只控制一个企业，其余企业由背景策略决定。
            action = agent.act(firm_state, eps)
            actions[agent.firm_id] = action
            next_state, rewards, done, _ = env.step(actions)
            reward = float(rewards[agent.firm_id, 0])
            agent.step(firm_state, action, reward, next_state[agent.firm_id], done)
            state = next_state
            score += reward

        eps = max(eps_end, eps_decay * eps)
        scores.append(score)
        if episode % int(cfg.get("log_every", 50)) == 0:
            avg = np.mean(scores[-int(cfg.get("log_every", 50)):])
            print(f"episode={episode} avg_score={avg:.2f} epsilon={eps:.3f}")
    return np.asarray(scores, dtype=np.float32)


def train_ppo(env: BeerGameEnv, agent: PPOAgent, cfg: dict):
    """Train a PPO agent for one target firm while other firms use a background policy."""
    rng = np.random.default_rng(cfg.get("seed", 42))
    background_policy = str(cfg.get("background_policy", "random"))
    background_target = int(cfg.get("background_base_stock_target", env.config.initial_inventory))
    base_seed = int(cfg.get("seed", 42))
    episodes = int(cfg.get("episodes", 500))
    rollout_episodes = int(cfg.get("rollout_episodes", 4))
    reward_scale = float(cfg.get("reward_scale", 1e-3))
    use_augmented_obs = bool(cfg.get("use_augmented_obs", False))

    # Reward-shaping options (see SRDQN feedback / externality shaping literature).
    use_feedback = bool(cfg.get("use_feedback_shaping", False))
    feedback_coef = float(cfg.get("feedback_coef", 1.0))
    use_externality = bool(cfg.get("use_externality_shaping", False))
    externality_coef = float(cfg.get("externality_coef", 1.0))
    use_smooth = bool(cfg.get("use_action_smoothing", False))
    smooth_coef = float(cfg.get("action_smooth_coef", 0.1))

    use_system_reward = bool(cfg.get("use_system_reward", False))

    use_relative_reward = bool(cfg.get("use_relative_reward", False))
    baseline_policy = str(cfg.get("baseline_policy", "base_stock"))
    baseline_target = int(cfg.get("baseline_target", env.config.initial_inventory))

    # Optional precomputed baseline value function for relative rewards.
    baseline_values: dict[int, float] | None = None
    if use_relative_reward:
        baseline_values = {}

    scores = []

    total_updates = max(1, episodes // rollout_episodes)
    agent.total_updates = total_updates

    for episode in range(1, episodes + 1):
        state = env.reset(seed=base_seed + episode)
        agent.reset_history()
        done = False
        score = 0.0
        ep_agent_total = 0.0
        ep_team_total = 0.0
        ep_others_cost = 0.0
        ep_length = 0
        last_action = None

        # For relative reward: rollout a baseline policy on the same demand stream.
        if use_relative_reward:
            baseline_env = BeerGameEnv(env.config)
            baseline_env.rng = np.random.default_rng(base_seed + episode)
            baseline_env.reset(seed=base_seed + episode)
            baseline_state = state.copy()
            baseline_reward_sum = 0.0
            baseline_step = 0

        while not done:
            obs_state = env.get_augmented_observation() if use_augmented_obs else state
            actions = make_background_actions(
                env,
                state,
                agent.firm_id,
                rng,
                background_policy=background_policy,
                target_inventory=background_target,
            )
            action = agent.act(obs_state[agent.firm_id], critic_state=obs_state.flatten())
            actions[agent.firm_id] = float(action)
            next_state, rewards, done, info = env.step(actions)
            raw_reward = float(rewards[agent.firm_id, 0])
            team_reward = float(rewards.sum())

            shaped_reward = team_reward * reward_scale if use_system_reward else raw_reward * reward_scale
            if use_smooth and last_action is not None:
                shaped_reward -= smooth_coef * abs(int(action) - int(last_action)) * reward_scale
            last_action = action

            if use_relative_reward:
                # Step baseline policy on a cloned environment with identical randomness.
                baseline_actions = make_background_actions(
                    baseline_env,
                    baseline_state,
                    agent.firm_id,
                    np.random.default_rng(base_seed + episode + baseline_step),
                    background_policy=baseline_policy,
                    target_inventory=baseline_target,
                )
                baseline_actions[agent.firm_id] = np.clip(
                    round(baseline_target - baseline_state[agent.firm_id, 2]), 0, env.config.max_order
                )
                baseline_next_state, baseline_rewards, baseline_done, _ = baseline_env.step(baseline_actions)
                baseline_reward_sum += float(baseline_rewards[agent.firm_id, 0])
                shaped_reward = (raw_reward - float(baseline_rewards[agent.firm_id, 0])) * reward_scale
                baseline_state = baseline_next_state
                baseline_step += 1

            agent.store_transition(shaped_reward, done)
            state = next_state
            score += raw_reward
            ep_agent_total += raw_reward
            ep_team_total += team_reward
            ep_length += 1

            if use_externality:
                demand = info["demand"]
                satisfied = info["satisfied_demand"]
                inventory = info["inventory"]
                h = env.config.holding_cost
                c = env.config.lost_sales_cost
                for j in range(env.num_firms):
                    if j == agent.firm_id:
                        continue
                    lost = max(float(demand[j]) - float(satisfied[j]), 0.0)
                    ep_others_cost += h * float(inventory[j]) + c * lost

        if use_relative_reward and ep_length > 0:
            baseline_values[episode] = baseline_reward_sum / ep_length

        if ep_length > 0:
            if use_feedback:
                # SRDQN-style feedback: reward the agent for the average profit of the
                # other supply-chain stages, aligning local learning with total profit.
                others_avg_per_step = (ep_team_total - ep_agent_total) / ep_length
                bonus_per_step = feedback_coef * others_avg_per_step * reward_scale
                agent.shape_last_episode(bonus_per_step, ep_length)
            if use_externality:
                # Penalize the agent for the true externalities it imposes on others
                # (holding + lost-sales costs), encouraging system-wide coordination.
                others_cost_avg = ep_others_cost / ep_length
                penalty_per_step = -externality_coef * others_cost_avg * reward_scale
                agent.shape_last_episode(penalty_per_step, ep_length)

        scores.append(score)

        if agent.should_update(done=True):
            obs_state = env.get_augmented_observation() if use_augmented_obs else state
            next_state_for_update = obs_state[agent.firm_id]
            next_critic_state_for_update = obs_state.flatten()
            loss_info = agent.update(next_state_for_update, next_critic_state_for_update)
            if episode % int(cfg.get("log_every", 50)) == 0:
                avg = np.mean(scores[-int(cfg.get("log_every", 50)):])
                loss_str = " ".join(f"{k}={v:.4f}" for k, v in loss_info.items())
                print(f"episode={episode} avg_score={avg:.2f} {loss_str}")

    # Flush any remaining rollouts
    if len(agent.states) > 0:
        obs_state = env.get_augmented_observation() if use_augmented_obs else state
        next_state_for_update = obs_state[agent.firm_id]
        next_critic_state_for_update = obs_state.flatten()
        agent.update(next_state_for_update, next_critic_state_for_update)

    return np.asarray(scores, dtype=np.float32), baseline_values


def train_ppo_best(
    env: BeerGameEnv,
    agent: PPOAgent,
    cfg: dict,
    policy_class=None,
) -> np.ndarray:
    """Train PPO and keep the checkpoint with the highest evaluation reward.

    Every ``eval_every`` episodes the current policy is evaluated on
    ``eval_episodes`` fresh episodes.  The best-performing checkpoint is saved
    and restored at the end of training, which often yields a more stable and
    higher final reward than simply using the last iterate.
    """
    rng = np.random.default_rng(cfg.get("seed", 42))
    background_policy = str(cfg.get("background_policy", "random"))
    background_target = int(
        cfg.get("background_base_stock_target", env.config.initial_inventory)
    )
    base_seed = int(cfg.get("seed", 42))
    episodes = int(cfg.get("episodes", 500))
    rollout_episodes = int(cfg.get("rollout_episodes", 4))
    reward_scale = float(cfg.get("reward_scale", 1e-3))
    use_feedback = bool(cfg.get("use_feedback_shaping", False))
    feedback_coef = float(cfg.get("feedback_coef", 1.0))
    use_externality = bool(cfg.get("use_externality_shaping", False))
    externality_coef = float(cfg.get("externality_coef", 1.0))
    use_smooth = bool(cfg.get("use_action_smoothing", False))
    smooth_coef = float(cfg.get("action_smooth_coef", 0.1))
    eval_every = int(cfg.get("eval_every", 50))
    eval_episodes = int(cfg.get("eval_episodes", 20))
    use_system_reward = bool(cfg.get("use_system_reward", False))

    use_augmented_obs = bool(cfg.get("use_augmented_obs", False))

    scores = []

    total_updates = max(1, episodes // rollout_episodes)
    agent.total_updates = total_updates

    best_path = Path(cfg.get("model_dir", "models/tmp")) / f"ppo_best_seed_{base_seed}.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    best_eval = -float("inf")

    class _PPOPolicy:
        def __init__(self, agent: PPOAgent):
            self.agent = agent

        def reset(self):
            self.agent.reset_history()

        def act(self, state: np.ndarray, firm_id: int) -> int:
            obs_state = env.get_augmented_observation() if use_augmented_obs else state
            return self.agent.eval_act(obs_state[firm_id])

    eval_policy = policy_class(agent) if policy_class is not None else _PPOPolicy(agent)

    for episode in range(1, episodes + 1):
        state = env.reset(seed=base_seed + episode)
        agent.reset_history()
        done = False
        score = 0.0
        ep_agent_total = 0.0
        ep_team_total = 0.0
        ep_others_cost = 0.0
        ep_length = 0
        last_action = None

        while not done:
            obs_state = env.get_augmented_observation() if use_augmented_obs else state
            actions = make_background_actions(
                env,
                state,
                agent.firm_id,
                rng,
                background_policy=background_policy,
                target_inventory=background_target,
            )
            action = agent.act(obs_state[agent.firm_id], critic_state=obs_state.flatten())
            actions[agent.firm_id] = float(action)
            next_state, rewards, done, info = env.step(actions)
            raw_reward = float(rewards[agent.firm_id, 0])
            team_reward = float(rewards.sum())

            shaped_reward = team_reward * reward_scale if use_system_reward else raw_reward * reward_scale
            if use_smooth and last_action is not None:
                shaped_reward -= smooth_coef * abs(int(action) - int(last_action)) * reward_scale
            last_action = action

            agent.store_transition(shaped_reward, done)
            state = next_state
            score += raw_reward
            ep_agent_total += raw_reward
            ep_team_total += team_reward
            ep_length += 1

            if use_externality:
                demand = info["demand"]
                satisfied = info["satisfied_demand"]
                inventory = info["inventory"]
                h = env.config.holding_cost
                c = env.config.lost_sales_cost
                for j in range(env.num_firms):
                    if j == agent.firm_id:
                        continue
                    lost = max(float(demand[j]) - float(satisfied[j]), 0.0)
                    ep_others_cost += h * float(inventory[j]) + c * lost

        if ep_length > 0:
            if use_feedback:
                others_avg_per_step = (ep_team_total - ep_agent_total) / ep_length
                bonus_per_step = feedback_coef * others_avg_per_step * reward_scale
                agent.shape_last_episode(bonus_per_step, ep_length)
            if use_externality:
                others_cost_avg = ep_others_cost / ep_length
                penalty_per_step = -externality_coef * others_cost_avg * reward_scale
                agent.shape_last_episode(penalty_per_step, ep_length)

        scores.append(score)

        if agent.should_update(done=True):
            obs_state = env.get_augmented_observation() if use_augmented_obs else state
            next_state_for_update = obs_state[agent.firm_id]
            next_critic_state_for_update = obs_state.flatten()
            loss_info = agent.update(next_state_for_update, next_critic_state_for_update)

            if episode % eval_every == 0 or episode == episodes:
                eval_result = evaluate_policy(
                    env,
                    eval_policy,
                    agent.firm_id,
                    eval_episodes,
                    seed=base_seed,
                )
                eval_mean = float(np.mean(eval_result["scores"]))
                if eval_mean > best_eval:
                    best_eval = eval_mean
                    agent.save(best_path)
                avg = np.mean(scores[-min(eval_every, len(scores)):])
                loss_str = " ".join(f"{k}={v:.4f}" for k, v in loss_info.items())
                print(
                    f"episode={episode} train_avg={avg:.2f} eval_mean={eval_mean:.2f} "
                    f"best={best_eval:.2f} {loss_str}"
                )

    if len(agent.states) > 0:
        obs_state = env.get_augmented_observation() if use_augmented_obs else state
        next_state_for_update = obs_state[agent.firm_id]
        next_critic_state_for_update = obs_state.flatten()
        agent.update(next_state_for_update, next_critic_state_for_update)

    if best_path.exists():
        agent.load(best_path)

    return np.asarray(scores, dtype=np.float32)


def train_ppo_vec(env: BeerGameEnv, agent: PPOAgent, cfg: dict):
    """Vectorized PPO training: parallel envs + batched network forward.

    This keeps the same total amount of environment interaction and gradient
    updates as ``train_ppo`` but reduces wall-clock time by stepping multiple
    environments in parallel and evaluating the policy on a batch of states.
    Transitions from each parallel environment are kept as contiguous
    trajectories so GAE is computed correctly.
    """
    rng = np.random.default_rng(cfg.get("seed", 42))
    background_policy = str(cfg.get("background_policy", "random"))
    background_target = int(
        cfg.get("background_base_stock_target", env.config.initial_inventory)
    )
    base_seed = int(cfg.get("seed", 42))
    episodes = int(cfg.get("episodes", 500))
    rollout_episodes = int(cfg.get("rollout_episodes", 4))
    reward_scale = float(cfg.get("reward_scale", 1e-3))
    num_envs = int(cfg.get("num_envs", 4))
    device = agent.device

    total_updates = max(1, episodes // rollout_episodes)
    agent.total_updates = total_updates

    env_vec = VectorizedBeerGameEnv(env.config, num_envs=num_envs, seed=base_seed)
    obs = env_vec.reset()

    # Per-environment trajectory buffers.
    ep_states: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
    ep_actions: list[list[int]] = [[] for _ in range(num_envs)]
    ep_log_probs: list[list[float]] = [[] for _ in range(num_envs)]
    ep_values: list[list[float]] = [[] for _ in range(num_envs)]
    ep_rewards: list[list[float]] = [[] for _ in range(num_envs)]
    ep_dones: list[list[bool]] = [[] for _ in range(num_envs)]
    ep_next_values: list[list[float]] = [[] for _ in range(num_envs)]

    ep_scores = np.zeros(num_envs, dtype=np.float32)
    all_scores: list[float] = []
    rollout_next_values: list[float] = []
    episodes_done = 0
    episodes_since_update = 0

    def flush_episode(env_id: int) -> None:
        nonlocal episodes_done, episodes_since_update
        agent.states.extend(ep_states[env_id])
        agent.actions.extend(ep_actions[env_id])
        agent.log_probs.extend(ep_log_probs[env_id])
        agent.values.extend(ep_values[env_id])
        agent.rewards.extend(ep_rewards[env_id])
        agent.dones.extend(ep_dones[env_id])
        rollout_next_values.extend(ep_next_values[env_id])

        ep_states[env_id].clear()
        ep_actions[env_id].clear()
        ep_log_probs[env_id].clear()
        ep_values[env_id].clear()
        ep_rewards[env_id].clear()
        ep_dones[env_id].clear()
        ep_next_values[env_id].clear()

        all_scores.append(float(ep_scores[env_id]))
        ep_scores[env_id] = 0.0
        episodes_done += 1
        episodes_since_update += 1

    while episodes_done < episodes:
        target_states = obs[:, agent.firm_id, :]
        with torch.no_grad():
            state_t = torch.FloatTensor(target_states).to(device)
            action_t, log_prob_t, value_t = agent.net.act(state_t)
        actions_arr = action_t.cpu().numpy().astype(np.int64)
        log_probs_arr = log_prob_t.cpu().numpy().astype(np.float32)
        values_arr = value_t.cpu().numpy().astype(np.float32)

        actions = make_background_actions_vec(
            env_vec,
            obs,
            agent.firm_id,
            rng,
            background_policy=background_policy,
            target_inventory=background_target,
        )
        actions[:, agent.firm_id] = actions_arr.astype(np.float32)

        next_obs, rewards_all, dones, _ = env_vec.step(actions)
        raw_rewards = rewards_all[:, agent.firm_id, 0]

        with torch.no_grad():
            next_target_states = torch.FloatTensor(next_obs[:, agent.firm_id, :]).to(
                device
            )
            next_vals = agent.net(next_target_states)[1].squeeze(-1).cpu().numpy()
        next_vals = next_vals.astype(np.float32)

        for i in range(num_envs):
            ep_states[i].append(target_states[i].copy())
            ep_actions[i].append(int(actions_arr[i]))
            ep_log_probs[i].append(float(log_probs_arr[i]))
            ep_values[i].append(float(values_arr[i]))
            ep_rewards[i].append(float(raw_rewards[i]) * reward_scale)
            done_i = bool(dones[i])
            ep_dones[i].append(done_i)
            ep_next_values[i].append(0.0 if done_i else float(next_vals[i]))
            ep_scores[i] += float(raw_rewards[i])

            if done_i:
                flush_episode(i)

        if episodes_since_update >= rollout_episodes and len(agent.states) > 0:
            next_values_arr = np.asarray(rollout_next_values, dtype=np.float32)
            loss_info = agent.update(next_values=next_values_arr)
            rollout_next_values.clear()
            episodes_since_update = 0
            log_every = int(cfg.get("log_every", 50))
            if episodes_done % log_every < num_envs or episodes_done == 0:
                window = min(log_every, len(all_scores))
                if window > 0:
                    avg = np.mean(all_scores[-window:])
                    loss_str = " ".join(f"{k}={v:.4f}" for k, v in loss_info.items())
                    print(f"episode={episodes_done} avg_score={avg:.2f} {loss_str}")

    # Flush any partial trajectories at the end.
    for i in range(num_envs):
        if ep_states[i]:
            flush_episode(i)

    if len(agent.states) > 0:
        next_values_arr = np.asarray(rollout_next_values, dtype=np.float32)
        agent.update(next_values=next_values_arr)

    return np.asarray(all_scores[:episodes], dtype=np.float32)


def train_a2c(env: BeerGameEnv, agent: A2CAgent, cfg: dict):
    """Train an A2C agent for one target firm while other firms use a background policy."""
    rng = np.random.default_rng(cfg.get("seed", 42))
    background_policy = str(cfg.get("background_policy", "random"))
    background_target = int(cfg.get("background_base_stock_target", env.config.initial_inventory))
    base_seed = int(cfg.get("seed", 42))
    episodes = int(cfg.get("episodes", 500))
    log_every = int(cfg.get("log_every", 50))
    scores = []

    for episode in range(1, episodes + 1):
        state = env.reset(seed=base_seed + episode)
        done = False
        score = 0.0

        while not done:
            actions = make_background_actions(
                env,
                state,
                agent.firm_id,
                rng,
                background_policy=background_policy,
                target_inventory=background_target,
            )
            action = agent.act(state[agent.firm_id])
            actions[agent.firm_id] = float(action)
            next_state, rewards, done, _ = env.step(actions)
            raw_reward = float(rewards[agent.firm_id, 0])
            agent.store_transition(raw_reward, done)
            state = next_state
            score += raw_reward

            if agent.should_update():
                next_state_for_update = state[agent.firm_id]
                agent.update(next_state_for_update)

        scores.append(score)

        if episode % log_every == 0:
            avg = np.mean(scores[-log_every:])
            print(f"episode={episode} avg_score={avg:.2f}")

    return np.asarray(scores, dtype=np.float32)


def train_sac(env: BeerGameEnv, agent: DiscreteSACAgent, cfg: dict):
    """Train a discrete SAC agent for one target firm while other firms use a background policy."""
    rng = np.random.default_rng(cfg.get("seed", 42))
    background_policy = str(cfg.get("background_policy", "random"))
    background_target = int(cfg.get("background_base_stock_target", env.config.initial_inventory))
    base_seed = int(cfg.get("seed", 42))
    episodes = int(cfg.get("episodes", 500))
    log_every = int(cfg.get("log_every", 50))
    scores = []

    for episode in range(1, episodes + 1):
        state = env.reset(seed=base_seed + episode)
        done = False
        score = 0.0

        while not done:
            actions = make_background_actions(
                env,
                state,
                agent.firm_id,
                rng,
                background_policy=background_policy,
                target_inventory=background_target,
            )
            action = agent.act(state[agent.firm_id])
            actions[agent.firm_id] = float(action)
            next_state, rewards, done, _ = env.step(actions)
            reward = float(rewards[agent.firm_id, 0])
            agent.step(state[agent.firm_id], action, reward, next_state[agent.firm_id], done)
            state = next_state
            score += reward

        scores.append(score)
        if episode % log_every == 0:
            avg = np.mean(scores[-log_every:])
            print(f"episode={episode} avg_score={avg:.2f}")

    return np.asarray(scores, dtype=np.float32)


def plot_training(
    scores: np.ndarray,
    output_path: str | Path,
    window: int = 50,
    title: str | None = None,
):
    setup_chinese_font()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores = np.asarray(scores, dtype=np.float32)

    if title is None:
        title = f"{output_path.stem.replace('_training_rewards', '').upper()} 训练奖励曲线"

    if scores.ndim == 1:
        moving = np.array([np.mean(scores[max(0, i - window + 1): i + 1]) for i in range(len(scores))])
        plt.figure(figsize=(9, 5))
        plt.plot(scores, alpha=0.35, label="单轮奖励")
        plt.plot(moving, label=f"{window}轮滑动平均")
    else:
        moving = np.array(
            [
                [np.mean(row[max(0, i - window + 1): i + 1]) for i in range(scores.shape[1])]
                for row in scores
            ]
        )
        mean_curve = moving.mean(axis=0)
        std_curve = moving.std(axis=0)
        x = np.arange(scores.shape[1])
        plt.figure(figsize=(9, 5))
        plt.plot(mean_curve, label=f"{window}轮滑动平均（多seed均值）")
        plt.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2, label="seed间标准差")

    plt.xlabel("训练轮次")
    plt.ylabel("奖励")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


DISPLAY_NAMES = {
    "random": "随机策略",
    "base_stock": "库存补足",
    "dqn": "DQN",
    "double_dqn": "Double DQN",
    "dueling_dqn": "Dueling DQN",
    "dueling_double_dqn": "Dueling Double DQN",
    "ppo": "PPO",
    "sac": "SAC",
    "random_background": "随机背景",
    "base_stock_background": "库存补足背景",
    "random_all": "全随机",
    "base_stock_all": "全库存补足",
    "single_agent_ddqn": "单智能体 DDQN",
    "multiagent_ddqn": "多智能体 DDQN",
}

COLORS = {
    "random": "#9aa0a6",
    "base_stock": "#6f7782",
    "dqn": "#4e79a7",
    "double_dqn": "#59a14f",
    "dueling_dqn": "#f28e2b",
    "dueling_double_dqn": "#e15759",
    "ppo": "#76b7b2",
    "sac": "#edc948",
    "random_background": "#4e79a7",
    "base_stock_background": "#e15759",
    "random_all": "#9aa0a6",
    "base_stock_all": "#6f7782",
    "single_agent_ddqn": "#4e79a7",
    "multiagent_ddqn": "#e15759",
}


def _collect_plot_data(results: dict, names: list[str]):
    labels = [DISPLAY_NAMES.get(name, name) for name in names]
    means = np.array([float(np.mean(results[name]["scores"])) for name in names])
    stds = np.array([float(np.std(results[name]["scores"])) for name in names])
    colors = [COLORS.get(name, "#4e79a7") for name in names]
    return labels, means, stds, colors


def _annotate_bars(ax, bars, means: np.ndarray):
    y_min, y_max = ax.get_ylim()
    span = max(y_max - y_min, 1.0)
    for bar, value in zip(bars, means):
        offset = 0.025 * span
        y = value + offset if value >= 0 else value - offset
        va = "bottom" if value >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{value:+.1f}",
            ha="center",
            va=va,
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
        )


def _annotate_horizontal_bars(ax, means: np.ndarray):
    x_min, x_max = ax.get_xlim()
    span = max(x_max - x_min, 1.0)
    for idx, value in enumerate(means):
        offset = 0.025 * span
        if value >= 0:
            x = value + offset
            ha = "left"
        else:
            x = min(value + 0.10 * span, -offset)
            ha = "left"
        ax.text(
            x,
            idx,
            f"{value:+.1f}",
            va="center",
            ha=ha,
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
        )


def _draw_horizontal_comparison(ax, results: dict, names: list[str], title: str):
    labels, means, stds, colors = _collect_plot_data(results, names)
    y = np.arange(len(names))
    ax.barh(y, means, xerr=stds, color=colors, alpha=0.9, capsize=4)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("平均评估奖励")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.invert_yaxis()

    x_min = float(np.min(means - stds))
    x_max = float(np.max(means + stds))
    span = max(x_max - x_min, 1.0)
    ax.set_xlim(x_min - 0.12 * span, x_max + 0.18 * span)
    _annotate_horizontal_bars(ax, means)


def _plot_vertical_comparison(results: dict, names: list[str], output_path: str | Path, title: str):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels, means, stds, colors = _collect_plot_data(results, names)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.92, capsize=5, width=0.62)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("平均评估奖励")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    y_min = float(np.min(means - stds))
    y_max = float(np.max(means + stds))
    span = max(y_max - y_min, 1.0)
    ax.set_ylim(y_min - 0.12 * span, y_max + 0.18 * span)
    _annotate_bars(ax, bars, means)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close()


def plot_policy_baseline_comparison(results: dict, output_path: str | Path):
    names = ["random", "base_stock", "dqn"]
    _plot_vertical_comparison(results, names, output_path, "基础策略对比：Random / Base-stock / DQN")


def plot_dqn_ablation_comparison(results: dict, output_path: str | Path):
    names = ["dqn", "double_dqn", "dueling_dqn", "dueling_double_dqn"]
    _plot_vertical_comparison(results, names, output_path, "DQN算法消融对比")


def plot_baseline_comparison(results: dict, output_path: str | Path):
    setup_chinese_font()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_names = ["random", "base_stock", "dqn", "double_dqn", "dueling_dqn", "dueling_double_dqn", "ppo"]
    dqn_names = ["dqn", "double_dqn", "dueling_dqn", "dueling_double_dqn", "ppo"]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        gridspec_kw={"width_ratios": [1.2, 1.0]},
    )
    _draw_horizontal_comparison(axes[0], results, all_names, "所有方法整体对比")
    _draw_horizontal_comparison(axes[1], results, dqn_names, "DQN与PPO局部放大")
    fig.suptitle("Baseline 与算法消融评估结果（误差线为标准差）", fontsize=15, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=220)
    plt.close()


def plot_background_policy_comparison(results: dict, output_path: str | Path):
    setup_chinese_font()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    method_names = ["random_background", "base_stock_background"]
    labels = [DISPLAY_NAMES[name] for name in method_names]
    score_arrays = [np.asarray(results[name]["scores"], dtype=np.float32) for name in method_names]
    means = np.array([float(np.mean(scores)) for scores in score_arrays])
    stds = np.array([float(np.std(scores)) for scores in score_arrays])
    colors = [COLORS[name] for name in method_names]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5.8),
        gridspec_kw={"width_ratios": [1.35, 0.85]},
    )

    ax = axes[0]
    box = ax.boxplot(
        score_arrays,
        patch_artist=True,
        widths=0.48,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "#222222", "markeredgecolor": "#222222", "markersize": 5},
        medianprops={"color": "#222222", "linewidth": 1.4},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.28)
        patch.set_edgecolor(color)
    for idx, scores in enumerate(score_arrays, start=1):
        jitter = np.linspace(-0.12, 0.12, len(scores))
        ax.scatter(
            np.full(len(scores), idx) + jitter,
            scores,
            s=22,
            color=colors[idx - 1],
            alpha=0.72,
            edgecolors="white",
            linewidths=0.35,
        )
    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels)
    ax.set_ylabel("单次评估 episode reward")
    ax.set_title("20 个评估 episode 分布")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for idx, (mean, std) in enumerate(zip(means, stds), start=1):
        ax.errorbar(
            idx,
            mean,
            yerr=std,
            fmt="o",
            color="#222222",
            capsize=5,
            markersize=4,
            linewidth=1.2,
        )
        ax.text(
            idx + 0.17,
            mean,
            f"均值 {mean:.1f}\nstd {std:.1f}",
            va="center",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
        )

    ax = axes[1]
    ax.axis("off")
    mean_delta = means[1] - means[0]
    std_delta = stds[1] - stds[0]
    std_reduction = (1 - stds[1] / stds[0]) * 100 if stds[0] != 0 else 0.0
    text = (
        "关键指标\n\n"
        f"随机背景均值：{means[0]:.2f}\n"
        f"库存补足背景均值：{means[1]:.2f}\n"
        f"均值差：{mean_delta:+.2f}\n\n"
        f"随机背景标准差：{stds[0]:.2f}\n"
        f"库存补足背景标准差：{stds[1]:.2f}\n"
        f"波动变化：{std_delta:+.2f}\n"
        f"波动下降：{std_reduction:.1f}%\n\n"
        "解读：均值接近，\n"
        "库存补足背景更稳定。"
    )
    ax.text(
        0.02,
        0.96,
        text,
        va="top",
        ha="left",
        fontsize=11,
        linespacing=1.55,
        bbox={"facecolor": "#f7f7f7", "edgecolor": "#d0d0d0", "boxstyle": "round,pad=0.55"},
    )

    fig.suptitle("其他企业随机背景 vs 库存补足背景", fontsize=15, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=220)
    plt.close()


def plot_multiagent_training(scores: np.ndarray, output_path: str | Path, window: int = 50):
    setup_chinese_font()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores = np.asarray(scores, dtype=np.float32)
    total_scores = scores.sum(axis=1)
    curves = [scores[:, i] for i in range(scores.shape[1])] + [total_scores]
    labels = [f"企业{i}" for i in range(scores.shape[1])] + ["全链路合计"]
    colors = ["#4e79a7", "#59a14f", "#f28e2b", "#e15759"]

    fig, ax = plt.subplots(figsize=(10, 5.6))
    for curve, label, color in zip(curves, labels, colors):
        moving = np.array([np.mean(curve[max(0, i - window + 1): i + 1]) for i in range(len(curve))])
        ax.plot(moving, label=f"{label} {window}轮滑动平均", color=color, linewidth=1.8)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_xlabel("训练轮次")
    ax.set_ylabel("episode reward")
    ax.set_title("多智能体训练奖励曲线")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close()


def plot_multiagent_eval_curve(eval_points: np.ndarray, eval_scores: np.ndarray, output_path: str | Path):
    setup_chinese_font()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    eval_points = np.asarray(eval_points, dtype=np.int32)
    eval_scores = np.asarray(eval_scores, dtype=np.float32)
    total_scores = eval_scores.sum(axis=1)
    curves = [eval_scores[:, i] for i in range(eval_scores.shape[1])] + [total_scores]
    labels = [f"企业{i}" for i in range(eval_scores.shape[1])] + ["全链路合计"]
    colors = ["#4e79a7", "#59a14f", "#f28e2b", "#e15759"]

    fig, ax = plt.subplots(figsize=(10, 5.6))
    for curve, label, color in zip(curves, labels, colors):
        ax.plot(eval_points, curve, marker="o", label=label, color=color, linewidth=1.9)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_xlabel("训练轮次")
    ax.set_ylabel("无探索评估 reward")
    ax.set_title("多智能体无探索评估曲线")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close()


def plot_multiagent_comparison(summary: dict, output_path: str | Path):
    setup_chinese_font()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    method_names = ["random_all", "base_stock_all", "single_agent_ddqn", "multiagent_ddqn"]
    labels = [DISPLAY_NAMES[name] for name in method_names]
    colors = [COLORS[name] for name in method_names]
    firm1_means = np.array([summary[name]["firm_1_mean_reward"] for name in method_names], dtype=np.float32)
    firm1_stds = np.array([summary[name]["firm_1_std_reward"] for name in method_names], dtype=np.float32)
    total_means = np.array([summary[name]["total_chain_mean_reward"] for name in method_names], dtype=np.float32)
    total_stds = np.array([summary[name]["total_chain_std_reward"] for name in method_names], dtype=np.float32)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1, 1]})

    for ax, means, stds, title in [
        (axes[0], firm1_means, firm1_stds, "目标企业1 reward 对比"),
        (axes[1], total_means, total_stds, "全链路 total reward 对比"),
    ]:
        x = np.arange(len(method_names))
        bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.92, width=0.62)
        ax.axhline(0, color="#333333", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_ylabel("平均评估奖励")
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        y_min = float(np.min(means - stds))
        y_max = float(np.max(means + stds))
        span = max(y_max - y_min, 1.0)
        ax.set_ylim(y_min - 0.12 * span, y_max + 0.20 * span)
        _annotate_bars(ax, bars, means)

    fig.suptitle("单智能体与多智能体策略对比", fontsize=15, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=220)
    plt.close()


def run_rule_baselines(env: BeerGameEnv, cfg: dict):
    firm_id = int(cfg["experiment"].get("firm_id", 1))
    eval_episodes = int(cfg["experiment"].get("eval_episodes", 20))
    target = int(cfg["baselines"].get("base_stock_target", env.config.initial_inventory))
    results = {}
    for name in ["random", "base_stock"]:
        policy = build_policy(name, env.config.max_order, seed=cfg["env"].get("seed", 42), target_inventory=target)
        results[name] = evaluate_policy(env, policy, firm_id, eval_episodes)
    return results
