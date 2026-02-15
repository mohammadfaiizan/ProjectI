"""
TF-Agents: environment, agent, policy, replay buffer concepts.
"""
import tensorflow as tf

def main():
    print("=" * 50)
    print("TF-Agents Reinforcement Learning")
    print("=" * 50)

    try:
        from tf_agents.environments import tf_py_environment
        from tf_agents.environments import suite_gym
        from tf_agents.agents.dqn import dqn_agent
        from tf_agents.networks import q_network
        from tf_agents.replay_buffers import tf_uniform_replay_buffer
        from tf_agents.policies import random_tf_policy
        from tf_agents.drivers import dynamic_step_driver
        from tf_agents.metrics import tf_metrics
        print("tf_agents imported successfully")
    except ImportError:
        print("tf_agents not installed. Install: pip install tf-agents")
        return

    print("\nEnvironment (CartPole):")
    env = suite_gym.load("CartPole-v1")
    tf_env = tf_py_environment.TFPyEnvironment(env)
    print(f"  Action spec: {tf_env.action_spec()}")
    print(f"  Observation spec: {tf_env.observation_spec()}")

    print("\nQ-Network and DQN Agent:")
    fc_layer_params = (100, 50)
    q_net = q_network.QNetwork(
        tf_env.observation_spec(),
        tf_env.action_spec(),
        fc_layer_params=fc_layer_params
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    train_step = tf.Variable(0)
    agent = dqn_agent.DqnAgent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=tf.keras.losses.Huber(),
        train_step_counter=train_step
    )
    agent.initialize()
    print(f"  Agent type: {type(agent).__name__}")

    print("\nReplay buffer:")
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=tf_env.batch_size,
        max_length=10000
    )
    print(f"  Max length: 10000")

    print("\nPolicy:")
    random_policy = random_tf_policy.RandomTFPolicy(tf_env.time_step_spec(), tf_env.action_spec())
    time_step = tf_env.reset()
    action_step = random_policy.action(time_step)
    print(f"  Random action: {action_step.action.numpy()}")

    print("\nTF-Agents demo complete.")

if __name__ == "__main__":
    main()
