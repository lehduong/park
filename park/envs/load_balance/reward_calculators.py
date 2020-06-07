from abc import ABC
from functools import reduce

OBJECTIVE_NAMES = ['makespan', 'waitingTime', 'completionTime']


class BaseRewardCalculator(ABC):
    """
        Base class of all reward calculators
        :param env: LoadBalanceEnv's instance
    """

    def __init__(self, env):
        self.env = env

    def get_reward(self, action):
        """
            :param action: int - index of assigned server
            :return: reward of that action
        """
        pass


class WaitingTimeReward(BaseRewardCalculator):
    """
        Optimizing the waiting time of each job
        Every time we assign a server for a job, reward equal to waiting time of that job\
            before it can execute
    """

    def get_reward(self, action):
        assigned_server = self.env.servers[action]
        curr_job = assigned_server.curr_job
        service_rate = assigned_server.service_rate

        # compute waiting time for processing job
        if curr_job is None or \
                curr_job.finish_time < self.env.wall_time.curr_time:
            wait_curr_job = 0
        else:
            wait_curr_job = curr_job.finish_time - self.env.wall_time.curr_time

        # compute waiting time for jobs in queue
        wait_queued_job = reduce(lambda acc, elem: acc + elem.size / service_rate,
                                 assigned_server.queue,
                                 0)

        return -1.0 * (wait_curr_job + wait_queued_job)


class CompletionTimeReward(WaitingTimeReward):
    """
        Optimize the completion time of each job 
        Everytime we assign a server for a job, return reward equal to completion time of\
            that job (i.e. waiting time + processing time on that server)
    """

    def get_reward(self, action):
        wait_time = super().get_reward(action)
        complete_time = wait_time + self.env.incoming_job.size / \
            self.env.servers[action].service_rate

        return -1.0 * complete_time


class MakespanReward(BaseRewardCalculator):
    """
        Directly optimize the makespan of all jobs arrival to the cluster
        IMPORTANT: Extremely sparse reward, only return makespan after finishing all jobs 
    """

    def get_reward(self, action):
        # TODO: The reward seem to be faulty because in the last state \
        # though a smaller makespan has higher reward than the longer one \
        # the final reward is smaller than default scarsity reward (i.e. 0)
        # done
        if self.env.incoming_job is None:
            return -1.0 * self.env.wall_time.curr_time
        else:
            return 0


class NormalizedMakespanReward(MakespanReward):
    """
        Normalized the makespan with total time to complete all job\
            the final reward is (total_time_to_complete_all_job) / make_span 

        This reward can be consider as the parallelism of scheduling (highest reward 
            being equal to num_servers if all servers have service rate of 1.0)
    """

    def get_reward(self, action):
        reward = super().get_reward(action)
        if reward != 0:
            norm = reduce(lambda acc, elem: acc + elem,
                          [job.size for job in self.env.finished_jobs],
                          0)
            return - 1.0 * norm / reward
        else:
            return reward


class RewardCalculator:
    """
        Wrapper of reward calculator
    """

    def __init__(self, objective, env):
        if not objective in OBJECTIVE_NAMES:
            raise ValueError("Expect objective name to be one of " + str(OBJECTIVE_NAMES) +
                             "but got: " + str(objective))
        if objective == 'makespan':
            self.reward_calculator = NormalizedMakespanReward(env)
        elif objective == 'waitingTime':
            self.reward_calculator = WaitingTimeReward(env)
        elif objective == 'completionTime':
            self.reward_calculator = CompletionTimeReward(env)

    def get_reward(self, act):
        return self.reward_calculator.get_reward(act)
