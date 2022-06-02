from collections import defaultdict


class Read_LOG(object):
    def __init__(self):
        self.start_step = []
        self.observation = defaultdict(str)
        self.action = defaultdict(str)
        self.sub_kg = defaultdict(list)
        self.bottleneck = defaultdict(list)

    def detect_new_game(self, line, step):
        """
        Detect the log, and see when is the new start.
        :param line: string.
        :return:
        """
        if "Copyright (c) 1981" in line:
            self.start_step.append(step)
        return True if "Copyright (c) 1981" in line else False

    def read_log(self, log_file_path):
        """
        REad the log file and explain all the actions.
        :param log_file_path:
        :return:
        """
        step = 0
        obs_sig = False
        subKG_sig = False
        with open(log_file_path, "r") as f:
            for line in f:
                if "==========" in line:
                    step = int(line.split()[-1].split("=")[0])
                self.detect_new_game(line, step)  # detect new start of the game
                if obs_sig:
                    self.observation[step] = line.replace('\n', '')
                    obs_sig = False
                if "Observation:" in line:
                    obs_sig = True
                if "Action" in line:
                    self.action[step] = line.split(": ")[-1].replace('\n', '')

                if subKG_sig:  # record the KG explaination
                    line.replace("\n", "")
                    if line.strip():
                        self.sub_kg[step].append(line.strip())
                    else:
                        subKG_sig = False

                if "SubKG Explanation:" in line:
                    subKG_sig = True

    def read_obs(self, obs_file_path):
        """
        Read bottlenecks
        :param obs_file_path:
        :return:
        """
        step, round = 0, 0
        explain_sig = False
        with open(obs_file_path, "r") as f:
            for line in f:
                if 'STEP:' in line:
                    step = int(line.split(':')[-1])
                    for idx, start in enumerate(self.start_step):
                        if step > start:
                            pass
                        else:
                            round = idx
                            break
                if explain_sig:
                    line.replace("\n", "")
                    if line.strip():
                        self.bottleneck[round][-1].append(line)
                    else:
                        explain_sig = False

                if ', because' in line:
                    self.bottleneck[round].append([line])
                    explain_sig = True


if __name__ == "__main__":

    reader = Read_LOG()
    reader.read_log("Q-BERT/qbert/logs/xRL_obs_sd7.txt")
    for step in reader.action:
        if step < 20:
            print(reader.action[step])
        # if 'window' in reader.action[step]:
        #     print(reader.action[step], '>', reader.sub_kg[step])
    # print(reader.start_step)
    reader.read_obs('Q-BERT/qbert/logs/xRL_sd7.txt')
    # print(reader.bottleneck)
