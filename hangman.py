import random
from functools import cache
import tensorflow as tf
import numpy as np


def num_chars(d):
    """
    get number of characters used in a list of words
    :param d: list of words
    :return: number of characters used in d
    """
    set_ = set()
    for s in d:
        for l_ in s:
            set_.add(l_)
    return len(set_), set_


@cache
def get_dataset(path):
    """
    load dataset of list of words
    :param path: path of the text file
    :return: list of words of the file in path
    """
    print("loading dataset")
    with open(path) as file:
        dataset = [s.strip() for s in file.readlines()]
    return dataset, *num_chars(dataset)


class Environment:
    """
    Environment used for hangman
    """
    def __init__(self, path, max_lives=10):
        self.dataset, self.letters_count, self.letters_list = get_dataset(path)
        self.letters_list = sorted(list(self.letters_list))
        # max length of words
        self.max_len = max([len(s) for s in self.dataset])
        # list of only the letters (a-z)
        self.only_letters_list = self.letters_list
        # list of letters and placeholder "_"
        self.letters_list = self.letters_list + ["_"]  # used for placeholder for guessed letters
        # max allowed lives
        self.max_lives = max_lives

        # current lives
        self.lives = 0
        # current considered word
        self.current_word = self.dataset[random.randrange(0, len(self.dataset))]
        # current considered word without the already guessed characters
        self.current_word_remaining = self.current_word
        # list of already guessed/used characters
        self.already_chosen_letters = []

    @property
    def current_word_guessed(self):
        """
        getter for the remaining part of the word to guess
        :return: string with "_" on the remaining letters
        """
        s = ""
        for i in range(len(self.current_word)):
            if self.current_word_remaining[i] == "_":
                s += self.current_word[i]
            else:
                s += "_"
        return s

    def act(self, letter: str):
        """
        act on the environemnt
        :param letter: letter proposed
        :return: (reward, done)
        """
        if letter in self.already_chosen_letters:
            raise Exception("already chosen")
        if letter not in self.only_letters_list:
            raise Exception(f"what? '{letter}'?")
        self.already_chosen_letters.append(letter)

        if letter not in self.current_word_remaining:
            self.lives += 1
            if self.lives == self.max_lives:
                self.reset()
                return -1, 1
            return 0, 0

        self.current_word_remaining = self.current_word_remaining.replace(letter, "_")
        if len(set(list(self.current_word_remaining))) == 1:
            self.reset()
            return 1, 1

        return 0, 0

    def reset(self):
        """
        reset environment / reinitialize
        """
        self.lives = 0
        self.current_word = self.dataset[random.randrange(0, len(self.dataset))]
        self.current_word_remaining = self.current_word
        self.already_chosen_letters = []


class Agent:
    def __init__(self, env: Environment, batch_size=256, discount=0.99, clip_eps=0.1, step_size=1e-4,
                 actor_rep=15, critic_rep=1):
        self.discount = discount
        self.clip_eps = clip_eps
        self.actor_rep = actor_rep
        self.critic_rep = critic_rep
        self.actor = tf.keras.Sequential([
            tf.keras.layers.Dense(64),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.tanh),
            tf.keras.layers.Dense(64),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.tanh),
            tf.keras.layers.Dense(len(env.only_letters_list), activation=tf.nn.softmax,
                                  kernel_initializer=tf.initializers.RandomNormal(stddev=0.005),
                                  bias_initializer=tf.initializers.RandomNormal(stddev=0.005))
        ])
        self.critic = tf.keras.Sequential([
            tf.keras.layers.Dense(64),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.tanh),
            tf.keras.layers.Dense(64),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.tanh),
            tf.keras.layers.Dense(1, activation="linear")
        ])
        self.optimizer_actor = tf.optimizers.legacy.Adam(step_size)
        self.optimizer_critic = tf.optimizers.legacy.Adam(step_size)
        self.batch_size = batch_size

    def mask(self, env: Environment):
        """
        get mask for the env
        :param env: env to consider
        :return: binary mask with 0 on already proposed letters, 1 otherwise
        """
        mask = np.ones(len(env.only_letters_list))
        for l_ in env.already_chosen_letters:
            mask[self.letter_to_int(env, l_)] = 0
        return mask

    def state(self, env: Environment):
        """
        state configuration for the agent
        :param env: environment to consider
        :return: a state representation for such env
        """
        word = np.zeros((env.max_len, len(env.letters_list)))
        for i in range(len(env.current_word_guessed)):
            word[i, self.letter_to_int(env, env.current_word_guessed[i])] = 1

        chosen = np.zeros(len(env.only_letters_list) + 1)
        for el in env.already_chosen_letters:
            chosen[self.only_letter_to_int(env, el)] = 1

        return np.concatenate((
            word.reshape((-1)),
            chosen.reshape((-1))
        ))

    @staticmethod
    def letter_to_int(env: Environment, letter):
        return env.letters_list.index(letter)

    @staticmethod
    def only_letter_to_int(env: Environment, letter):
        return env.only_letters_list.index(letter)

    @staticmethod
    def int_to_only_letter(env: Environment, letter_idx):
        return env.only_letters_list[letter_idx]

    def learn(self, states, new_states, samples, rewards, dones, masks):
        """
        Proximal Policy Optimization (PPO) implementation using TD(0)
        """
        rewards = np.reshape(rewards, (-1, 1))
        dones = np.reshape(dones, (-1, 1))
        actions = tf.one_hot(samples, depth=masks.shape[-1]).numpy()
        initial_probs = None
        val = self.critic(states)
        new_val = self.critic(new_states)
        reward_to_go = tf.stop_gradient(rewards + self.discount * new_val * (1-dones))
        td_error = (reward_to_go - val).numpy()

        for _ in range(self.actor_rep):
            indexes = np.random.choice(range(0, len(states)), min(self.batch_size, len(states)), replace=False)
            with tf.GradientTape() as a_tape:
                probs = self.actor(states[indexes])
                probs = probs * masks[indexes]
                probs = probs / tf.reduce_sum(probs, axis=-1, keepdims=True)
                selected_actions_probs = tf.reduce_sum(probs * actions[indexes], axis=-1, keepdims=True)
                if initial_probs is None: initial_probs = tf.convert_to_tensor(tf.stop_gradient(selected_actions_probs))
                importance_sampling_ratio = selected_actions_probs / initial_probs
                loss_actor = tf.minimum(
                    td_error[indexes] * importance_sampling_ratio,
                    td_error[indexes] * tf.clip_by_value(importance_sampling_ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                )
                loss_actor = tf.reduce_mean(-loss_actor)

            grad_actor = a_tape.gradient(loss_actor, self.actor.trainable_weights)
            self.optimizer_actor.apply_gradients(zip(grad_actor, self.actor.trainable_weights))

        for _ in range(self.critic_rep):
            indexes = np.random.choice(range(0, len(states)), min(self.batch_size, len(states)), replace=False)
            with tf.GradientTape() as c_tape:
                val = self.critic(states[indexes])
                new_val = tf.stop_gradient(self.critic(new_states[indexes]))
                reward_to_go = tf.stop_gradient(rewards[indexes] + self.discount * new_val * (1-dones[indexes]))
                loss_critic = tf.losses.mean_squared_error(val, reward_to_go)[:, None]
                loss_critic = tf.reduce_mean(loss_critic)
            grad_critic = c_tape.gradient(loss_critic, self.critic.trainable_weights)
            self.optimizer_critic.apply_gradients(zip(grad_critic, self.critic.trainable_weights))

