import gym
import argparse
import tensorflow as tf
from dqn import DQN

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--episodes', type=int, default=1000, help='No of episodes')
    parser.add_argument('--episode_len', type=int, default=500, help='length of episode')
    parser.add_argument('--openai_env', type=str, required=True, help='env like MountainCar-v0, CartPole-v0 etc')
    parser.add_argument('--epsilon', type=float, default=1, help='exploration parameter')
    parser.add_argument('--epsilon_decay', type=float, default=0.995, help='epsilon decay rate')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning_rate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')

    args = parser.parse_args()
    parameters = {}
    for key, value in vars(args).items():
        parameters[key] = value

    env = gym.make(args.openai_env)
    model = DQN(env, parameters)
    model.build_model()
    saver = tf.train.Saver(max_to_keep=1)

    total_reward = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(args.episodes):
            curr_state = env.reset().reshape(-1, env.observation_space.shape[0])
            j = 0
            done = False

            if model.epsilon > 0.15:
                model.epsilon *= model.epsilon_decay
                print(model.epsilon)

            #for j in range(args.episode_len):
            while done==False:
                print("episode:{} trial:{}".format(i, j))
                env.render()
                _, action = model.act(sess, curr_state)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                print("action:{} next_state:{} ".format(action, next_state))

                next_state = next_state.reshape(-1, env.observation_space.shape[0])
                model.add_to_memory(curr_state, action, reward, next_state, done)
                model.replay(sess)

                curr_state = next_state
                j += 1
            if j < 199:
                print("Comleted in {} episodes".format(i))
                saver.save(sess, "checkpoint/ckpt-" + str(i), write_meta_graph=False)
                break
            else:
                saver.save(sess, "checkpoint/ckpt-" + str(i), write_meta_graph=False)

if __name__ == '__main__':
    main()
