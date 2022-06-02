from nltk_gen import nltk_gen
from GPT2_gen import GPT_2_gen
import torch
import sentencepiece as spm
import numpy as np


class XRL(object):
    def __init__(self, file_path, tsv_file, args, obs_file_path, spm_file, kg_file_dir):
        self.x_action = "The action we take is "
        self.x_file = open(file_path, "w")
        self.obs_file = open(obs_file_path, "w")
        self.tsv_file = tsv_file
        self.dirs = [
            "north",
            "south",
            "east",
            "west",
            "southeast",
            "southwest",
            "northeast",
            "northwest",
        ]
        self.nltk = nltk_gen()
        self.GPT_2_gen = GPT_2_gen(args)
        self.bottleneck_graph = None

        # decode
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(spm_file)

        # save KG
        self.kg_file_dir = kg_file_dir

    def location_split(self, node):
        if "house" in node and "_" in node:
            return " ".join(node.split("_"))
        else:
            return "in the " + node

    def clear_log(self):
        self.x_file.truncate(0)
        self.obs_file.truncate(0)

    def graph_explain(self, file, graph_num, entity_1, entity_2, front_sig=False):
        if entity_2 == entity_1:
            return False
        if front_sig:
            file.write("; and ")
        if graph_num == 0:
            if entity_1 == "you":
                file.write("I am " + entity_2)
            else:
                file.write(entity_1 + " is " + entity_2)
            return True

        elif graph_num == 1:
            if entity_1 == "you":
                if "_house" in entity_2:
                    file.write("I am " + entity_2)
                else:
                    file.write("I have " + entity_2)
            else:
                if entity_1 in self.dirs:
                    file.write(
                        "there is "
                        + self.location_split(entity_2)
                        + " in the "
                        + entity_1
                    )
                else:
                    if "_house" in entity_2:
                        file.write(entity_1 + " is " + self.location_split(entity_2))
                    else:
                        file.write(entity_1 + " has " + entity_2)
            return True

        elif graph_num == 2:  # location
            # print('entity_2 is ', entity_2, self.nltk.adj_checker(entity_2))
            if not self.nltk.adj_checker(entity_2):
                if entity_1 == "you":
                    file.write("I am " + self.location_split(entity_2) + " now")
                else:
                    file.write(entity_1 + " is " + self.location_split(entity_2))
                return True
            else:
                return False

        elif graph_num == 3:
            if entity_1 == entity_2:
                return False
            else:
                file.write(entity_1 + " others " + entity_2)
                return True

        elif graph_num == 4:
            pass
            # if self.nltk.adj_checker(entity_2):
            #     file.write(entity_1 + ' is ' + entity_2)
            #
            # elif entity_2 in self.dirs or 'house' in entity_2:
            #         if entity_1 == 'you':
            #             file.write('I am ' + self.location_split(entity_2) + ' now')
            #         else:
            #             file.write(entity_1 + ' is ' + self.location_split(entity_2))
            # elif self.nltk.noun_checker(' '.join(entity_1.split('_'))) and self.nltk.noun_checker(' '.join(entity_2.split('_'))):
            #     file.write(' '.join(entity_1.split('_')) + ' has ' + ' '.join(entity_2.split('_')))
            # elif 'you' in entity_1:
            #     sentences = ['you are in the ' + ' '.join(entity_2.split('_')),
            #                  'you have ' + ' '.join(entity_2.split('_'))]
            #     probs = self.GPT_2_gen.calculate_prob(sentences=sentences)
            #     file.write(sentences[probs.index(min(probs))])
            # else:
            #     file.write(entity_1 + ' 444 ' + entity_2)
            # return True
        return False

        # elif graph_num == 4:

    def explain_bottleneck(self, file, graph_infos_prev, graph_infos, action, obs_before):
        """
        compare the difference and show here.
        :param graph_info:
        :return:
        """
        file.write("=========bottleneck is overcomed========" + "\n")
        file.write('Action:' + action + '.' + 'Location:' + obs_before.split('\n')[0] + "\n")
        explaination = self.adj_to_entity(
            torch.nonzero(
                torch.tensor(
                    (
                        graph_infos_prev[0].graph_state_rep[1]
                        - graph_infos[0].graph_state_rep[1]
                    )
                    != 0
                )
            )
        )
        print("explaination is ", explaination)
        print("Action is =>, ", action)
        file.write("Action is -> " + action + "\n")

        # file.write("Difference is -> " + action + "\n")
        # for (ent_1, ent_2) in explaination:
        #     file.write(ent_1 + " - " + ent_2 + "\n")

        # KG triples
        file.write("\n")
        file.write("KG diff are: \n")

        graph_state = [g.graph_state for g in graph_infos]
        # print('!!!', graph_state[0].edges, '\n')
        edge_labels = {
            e: graph_state[0].edges[e]["rel"] for e in graph_state[0].edges
        }

        graph_state_prev = [g.graph_state for g in graph_infos_prev]
        # print(graph_state_prev[0])
        edge_labels_prev = {
            e: graph_state_prev[0].edges[e]["rel"] for e in graph_state_prev[0].edges
        }
        triples = set()
        for k, v in edge_labels_prev.items():
            triples.add((k[0], v, k[1]))

        triples_more = []
        for k, v in edge_labels.items():
            if(k[0], v, k[1]) not in triples:
                triples_more.append((k[0], v, k[1]))
        for triple in triples_more:
            file.write(' '.join(triple))
            file.write("\n")

    def graph_explain_info(self, file, graph_num, entity, graph_info, front_sig):
        print('---', graph_num, entity)
        # if front_sig:
        #     file.write('; and ')
        if graph_num == 0:  # __ is __
            graph_state = [g.graph_state_1 for g in graph_info][0].edges
            for edge in graph_state:
                if (
                    edge[0].lower()!=edge[1].lower() and
                    (entity.lower() == edge[0].lower() or entity.lower() == edge[1].lower())
                    and edge[0].lower() in self.tsv_file[1].keys()
                    and edge[1].lower() in self.tsv_file[1].keys()
                ):
                    print('edges =>', edge[0], edge[1], edge[1] in self.tsv_file[1].keys())
                    file.write(edge[0] + " is " + edge[1] + "\n")
                    # file.write(str(graph_state) + "\n")
            return True

        elif graph_num == 1:  # you have __
            graph_state = [g.graph_state_2 for g in graph_info][0].edges
            file.write("I have " + entity + "\n")
            print("I have " + entity + "\n")
            return True

        elif graph_num == 2:  # __ in __
            graph_state = [g.graph_state_3 for g in graph_info][0].edges
            for edge in graph_state:
                print('***', entity, edge)
                print(entity.lower() == edge[0].lower(), entity.lower() == edge[1].lower(), edge[0].lower() in self.tsv_file[1].keys(),edge[1].lower() in self.tsv_file[1].keys())
                if (
                    edge[0].lower() != edge[1].lower() and
                    entity.lower() == edge[0].lower()
                    or entity.lower() == edge[1].lower()
                    and edge[0].lower() in self.tsv_file[1].keys()
                    and edge[1].lower() in self.tsv_file[1].keys()
                ):
                    if edge[0] == "you":
                        file.write("I am " + self.location_split(edge[1]) + " now. \n")
                    else:
                        file.write(
                            edge[0] + " is " + self.location_split(edge[1]) + "\n"
                        )
                    print(edge[0] + " is " + self.location_split(edge[1]))
            return True

        elif graph_num == 3:  # others
            graph_state = [g.graph_state_3 for g in graph_info][0].edges
            for edge in graph_state:
                print('***', entity, graph_state[edge[0], edge[1]]['rel'])
                print(entity.lower() == edge[0].lower(), entity.lower() == edge[1].lower(),
                      edge[0].lower() in self.tsv_file[1].keys(), edge[1].lower() in self.tsv_file[1].keys())
                if (
                        edge[0].lower() != edge[1].lower() and
                        entity.lower() == edge[0].lower()
                        or entity.lower() == edge[1].lower()
                        and edge[0].lower() in self.tsv_file[1].keys()
                        and edge[1].lower() in self.tsv_file[1].keys()
                ):
                    file.write(
                        edge[0] + graph_state[edge[0], edge[1]]['rel'] + self.location_split(edge[1]) + "\n"
                    )
            return True


    def adj_to_entity(self, adj):
        diff = []
        for idx in adj:
            entity_1_id = int(idx[0].cpu().numpy())
            entity_2_id = int(idx[1].cpu().numpy())
            entity_1_node = self.tsv_file[0][entity_1_id]
            entity_2_node = self.tsv_file[0][entity_2_id]
            diff.append((entity_1_node, entity_2_node))
        return diff

    def forward(
        self,
        step,
        action,
        output_gat,
        reward,
        obs_before,
        obs_after,
        graph_infos_prev,
        graph_infos_now,
        bottleneck=False,
        reward_change=True,
        value=None,
        score=0,
        place_prev=None,
    ):
        #check whether the game restart
        self.x_file.write('=' * 20 + 'NEW' + '='*20 + '\n')
        self.x_file.write('STEP:' + str(step) + '\n')
        if bottleneck:  # save bottleneck graph here.
            self.bottleneck_graph = graph_infos_now
            self.x_file.write('Bottleneck happens \n')
        else:
            self.x_file.write('Score is' + str(score) + '\n')
            if not reward_change:  # False False
                self.explain_bottleneck(
                    file=self.x_file,
                    graph_infos_prev=graph_infos_prev,
                    graph_infos=graph_infos_now,
                    action=action,
                    obs_before=obs_before,
                )
            else:
                "====not bottleneck, reward changed===="
                if value:
                    self.x_file.write('High value:' + str(value) + '\n')

                else:
                    self.x_file.write('Reward changed \n')
                # self.x_file.write(str(graph_infos_now[0].graph_state_rep[0]))
            self.x_file.write("=" * 20 + "\n")
            self.x_file.write("Reward is " + str(reward) + "\n")
            self.x_file.write("Observation before action is =>" + obs_before + "\n")
            self.x_file.write("Observation after action is =>" + obs_after + "\n")

            # expalin action here.
            if action in self.dirs:
                self.x_file.write("I choose to go to " + action + ", because" + "\n")
            elif action in ["up", "down"]:
                self.x_file.write("I choose to go " + action + ", because" + "\n")
            else:
                self.x_file.write("I " + action + ", because" + "\n")

            value_order = []
            rels = ["is", "have", "in", "others", "all"]
            front_sig = False

            # consider every subgraph
            for idx, (value, adj) in reversed(list(enumerate(output_gat))):
                # print(idx, ' =>', value)
                # print(idx, ' =>', adj)
                print('value.indices', value.indices)
                for i in range(3):  # consider top 3
                    value_node = int(value.indices[i].cpu().numpy())  # node's id
                    node_value = value.values[i].cpu().detach().numpy()  # node's value
                    print(value_node, node_value)
                    if self.tsv_file[0][value_node] != 'you' and node_value > -100:
                        if self.graph_explain_info(
                            file=self.x_file,
                            graph_num=idx,
                            entity=self.tsv_file[0][value_node],
                            graph_info=graph_infos_prev,
                            front_sig=front_sig,
                        ):
                            front_sig = True
                        # break
                        # for loc in range(adj.shape[0]):
                        #     adj_node = int(adj[loc][0].cpu().numpy())
                        #     adj_node_obj = int(adj[loc][1].cpu().numpy())
                        #     # print('adj =>', adj_node, adj_node_obj)
                        #     if value_node == adj_node:
                        #         value_order.append([node_value,
                        #                             self.tsv_file[0][value_node] + ' ' + rels[idx] + ' ' + self.tsv_file[0][
                        #                                 adj_node_obj]])
                        #         if i == 0:
                        #             if self.graph_explain(graph_num=idx,
                        #                                    entity_1=self.tsv_file[0][value_node],
                        #                                    entity_2=self.tsv_file[0][adj_node_obj],
                        #                                    front_sig=front_sig):
                        #                 front_sig = True
                        # print(idx, 'graph', '!!!REASON =>', self.tsv_file[0][value_node], rels[idx],
                        #       self.tsv_file[0][adj_node_obj])
            # if value_order:
            #     value_order.sort(key=lambda x: x[0])
            #     print(value_order[-1][-1])
            #     if len(value_order) > 1:
            #         print(value_order[-2][-1])
            #
            # else:
            #     print('Reason cannot find in the subgraph!')
            #     print(output_gat)
            # print('=' * 20)
            self.x_file.write("\n")

    def log(
        self,
        obs_reps,
        action,
        bottleneck,
        step,
        graph_info,
        reward,
        output_gat,
        query_important,
        scores_after,
        obs_next,
    ):
        """
        A txt file with (1) observations, (2) action, whether this is bottleneck
        :param obs_reps: numpy. batch_size * 4 * obs
        :param action:
        :param bottleneck:
        :return:
        """
        self.obs_file.write("=" * 10 + "STEP " + str(step) + "=" * 10 + "\n")

        # log bottleneck
        self.obs_file.write("Bottleneck " + str(bottleneck) + "\n")  # log bottleneck

        # log obs
        self.obs_file.write("Observation: \n")
        for index in range(4):  # write the obs
            self.obs_file.write(
                self.sp.decode_ids([i for i in obs_reps[0][index].tolist() if i != 0])
                + "\n"
            )
        self.obs_file.write("\n")
        # log action
        self.obs_file.write("Action: " + action + "\n")

        # log obs_next
        self.obs_file.write("Observation next: \n")
        for index in range(4):  # write the obs
            self.obs_file.write(
                self.sp.decode_ids([i for i in obs_next[0][index].tolist() if i != 0])
                + "\n"
            )
        self.obs_file.write("\n")
        # log reward
        self.obs_file.write("Reward is " + str(reward) + "\n")
        # log KG
        np.save(
            self.kg_file_dir + "/step_" + str(step) + ".npy",
            graph_info[0].graph_state_rep[1],
        )

        # explanation
        self.obs_file.write("\n")
        self.obs_file.write("Obs Explanation: \n")
        for i in range(4):
            self.obs_file.write(str(self.sp.decode_ids(query_important[i])) + "\n")
        self.obs_file.write("\n")
        self.obs_file.write("SubKG Explanation: \n")
        front_sig = False
        for idx, (value, adj) in reversed(list(enumerate(output_gat))):
            for i in range(3):  # consider top 1
                value_node = int(value.indices[i].cpu().numpy())  # node's id
                node_value = value.values[i].cpu().detach().numpy()  # node's value
                print('(((', idx, value_node, node_value, self.tsv_file[0][value_node])
                if node_value > -100:
                    print(')))', idx, value_node, node_value, self.tsv_file[0][value_node])
                    if self.graph_explain_info(
                        file=self.obs_file,
                        graph_num=idx,
                        entity=self.tsv_file[0][value_node],
                        graph_info=graph_info,
                        front_sig=front_sig,
                    ):
                        front_sig = True
                    # break

        # KG triples
        self.obs_file.write("\n")
        self.obs_file.write("KG are: \n")
        graph_state = [g.graph_state for g in graph_info]
        edge_labels = {
            e: graph_state[0].edges[e]["rel"] for e in graph_state[0].edges
        }
        triples = []
        for k, v in edge_labels.items():
            triples.append((k[0], v, k[1]))
        self.obs_file.write(str(triples))
        self.obs_file.write("\n")

        # scores
        self.obs_file.write("Scores: " + str(scores_after[0]) + '\n')