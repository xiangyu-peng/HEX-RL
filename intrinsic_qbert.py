import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import os
from os.path import basename, splitext
import numpy as np
import time
import sentencepiece as spm
from statistics import mean

from jericho import *
from jericho.template_action_generator import TemplateActionGenerator
from jericho.util import unabbreviate, clean
import jericho.defines
import copy

# from representations import StateAction
from models import QBERT
from env import *
from vec_env import *
import logger

import random

# from extraction import kgextraction
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
device = torch.device("cuda")

# explain RL
from xRL_templates import XRL


def configure_logger(log_dir):
    logger.configure(log_dir, format_strs=["log"])
    global tb
    tb = logger.Logger(
        log_dir,
        [
            logger.make_output_format("tensorboard", log_dir),
            logger.make_output_format("csv", log_dir),
            logger.make_output_format("stdout", log_dir),
        ],
    )
    global log
    logger.set_level(60)
    log = logger.log


class QBERTTrainer(object):
    """

    QBERT main class.


    """

    def __init__(self, params, args):
        print("----- Initiating ----- ")
        print("----- step 1 configure logger")
        self.seed = params["seed"]
        torch.manual_seed(params["seed"])
        np.random.seed(params["seed"])
        random.seed(params["seed"])
        configure_logger(params["output_dir"])
        log("Parameters {}".format(params))
        self.params = params
        self.chkpt_path = os.path.dirname(self.params["checkpoint_path"])
        if not os.path.exists(self.chkpt_path):
            os.mkdir(self.chkpt_path)
        print("----- step 2 load pre-collected things")
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(params["spm_file"])
        # askbert_args = {'input_text': '', 'length': 10, 'batch_size': 1, 'temperature': 1, 'model_name': '117M',
        #                'seed': 0, 'nsamples': 10, 'cutoffs': "6.5 -7 -5", 'write_sfdp': False, 'random': False}
        # self.extraction = kgextraction.World([], [], [], askbert_args)
        self.askbert = params["extraction"]
        print("----- step 3 build QBERTEnv")
        kg_env = QBERTEnv(
            rom_path=params["rom_file_path"],
            seed=params["seed"],
            spm_model=self.sp,
            tsv_file=params["tsv_file"],
            attr_file=params["attr_file"],
            step_limit=params["reset_steps"],
            stuck_steps=params["stuck_steps"],
            gat=params["gat"],
            askbert=self.askbert,
            clear_kg=params["clear_kg_on_reset"],
            subKG_type=params["subKG_type"],
        )
        self.vec_env = VecEnv(
            num_envs=params["batch_size"],
            env=kg_env,
            openie_path=params["openie_path"],
            redis_db_path=params["redis_db_path"],
            buffer_size=params["buffer_size"],
            askbert=params["extraction"],
            training_type=params["training_type"],
            clear_kg=params["clear_kg_on_reset"],
        )
        env = FrotzEnv(params["rom_file_path"])
        # self.binding = env.bindings
        self.binding = jericho.load_bindings(params["rom_file_path"])
        self.max_word_length = self.binding["max_word_length"]
        self.template_generator = TemplateActionGenerator(self.binding)
        print("----- step 4 build FrotzEnv and templace generator")
        self.max_game_score = env.get_max_score()
        self.cur_reload_state = env.get_state()
        self.vocab_act, self.vocab_act_rev = load_vocab(env)
        print("----- step 5 build Qbert model")
        self.model = QBERT(
            params,
            self.template_generator.templates,
            self.max_word_length,
            self.vocab_act,
            self.vocab_act_rev,
            len(self.sp),
            gat=self.params["gat"],
            argmax_sig=self.params["argmax_sig"]
        ).cuda()
        print("----- step 6 set training parameters")
        self.batch_size = params["batch_size"]
        if params["preload_weights"]:
            print("preload_weights are => ", params["preload_weights"])
            self.model = torch.load(self.params["preload_weights"])["model"]
        self.optimizer = optim.Adam(self.model.parameters(), lr=params["lr"])

        self.loss_fn1 = nn.BCELoss()
        self.loss_fn2 = nn.BCEWithLogitsLoss()
        self.loss_fn3 = nn.MSELoss()

        self.chained_logger = params["chained_logger"]
        self.total_steps = 0

        #####BBB#####
        self.early_stop_score = params["early_stop_score"]
        self.early_stop_score_count = params["early_stop_score_count"]
        self.eval_mode = params["eval_mode"]
        print("----- Init finished! ----- ")

        self.tsv_file = self.load_vocab_kge(params["tsv_file"])

        # explain RL
        self.random_action = params["random_action"]
        # print('self.random_action', self.random_action)
        if self.random_action:
            xRL_file_path = "/Q-BERT/qbert/logs/xRL_sd" + str(self.seed) + "_random_" + params['game_name'] +".txt"
            xRL_obs_file_path = "/Q-BERT/qbert/logs/xRL_obs_sd" + str(self.seed) + "_random_" + params['game_name'] +".txt"
            xRL_kg_file_dir = "/Q-BERT/qbert/logs/KG_sd" + str(self.seed) + "_random" + "_" + params['game_name']
        else:
            xRL_file_path = "/Q-BERT/qbert/logs/xRL_sd" + str(self.seed) + "_" + params['game_name'] +".txt"
            xRL_obs_file_path = "/Q-BERT/qbert/logs/xRL_obs_sd" + str(self.seed) + "_" + params['game_name'] +".txt"
            xRL_kg_file_dir = "/Q-BERT/qbert/logs/KG_sd" + str(self.seed) + "_" + params['game_name']
        if not os.path.exists(xRL_kg_file_dir):
            os.makedirs(xRL_kg_file_dir)
        self.XRL = XRL(
            file_path=xRL_file_path,
            tsv_file=self.tsv_file,
            args=args,
            obs_file_path=xRL_obs_file_path,
            spm_file=params["spm_file"],
            kg_file_dir=xRL_kg_file_dir,
        )

        self.random_action = False

    def log_file(self, str):
        with open(self.chained_logger, "a+") as fh:
            fh.write(str)

    def generate_targets(self, admissible, objs):
        """
        Generates ground-truth targets for admissible actions.

        :param admissible: List-of-lists of admissible actions. Batch_size x Admissible
        :param objs: List-of-lists of interactive objects. Batch_size x Objs
        :returns: template targets and object target tensors

        """
        tmpl_target = []
        obj_targets = []
        for adm in admissible:
            obj_t = set()
            cur_t = [0] * len(self.template_generator.templates)
            for a in adm:
                cur_t[a.template_id] = 1
                obj_t.update(a.obj_ids)
            tmpl_target.append(cur_t)
            obj_targets.append(list(obj_t))
        tmpl_target_tt = torch.FloatTensor(tmpl_target).cuda()

        # Note: Adjusted to use the objects in the admissible actions only
        object_mask_target = []
        for objl in obj_targets:  # in objs
            cur_objt = [0] * len(self.vocab_act)
            for o in objl:
                cur_objt[o] = 1
            object_mask_target.append([[cur_objt], [cur_objt]])
        obj_target_tt = torch.FloatTensor(object_mask_target).squeeze().cuda()
        return tmpl_target_tt, obj_target_tt

    def generate_graph_mask(self, graph_infos):
        assert len(graph_infos) == self.batch_size
        mask_all = []
        # TODO use graph dropout for masking here
        for graph_info in graph_infos:
            mask = [0] * len(self.vocab_act.keys())
            # Case 1 (default): KG as mask
            if self.params["masking"] == "kg":
                # Uses the knowledge graph as the mask.
                graph_state = graph_info.graph_state
                ents = set()
                for u, v in graph_state.edges:
                    ents.add(u)
                    ents.add(v)
                    # print('graph-mask', u,v)
                # Build mask: only use those related to entities
                for ent in ents:
                    for ent_word in ent.split():
                        if ent_word[: self.max_word_length] in self.vocab_act_rev:
                            idx = self.vocab_act_rev[ent_word[: self.max_word_length]]
                            mask[idx] = 1
                if self.params["mask_dropout"] != 0:
                    drop = random.sample(
                        range(0, len(self.vocab_act.keys()) - 1),
                        int(self.params["mask_dropout"] * len(self.vocab_act.keys())),
                    )
                    for i in drop:
                        mask[i] = 1
            # Case 2: interactive objects ground truth as the mask.
            elif self.params["masking"] == "interactive":
                # Uses interactive objects grount truth as the mask.
                for o in graph_info.objs:
                    o = o[: self.max_word_length]
                    if o in self.vocab_act_rev.keys() and o != "":
                        mask[self.vocab_act_rev[o]] = 1
                    if self.params["mask_dropout"] != 0:
                        drop = random.sample(
                            range(0, len(self.vocab_act.keys()) - 1),
                            int(
                                self.params["mask_dropout"] * len(self.vocab_act.keys())
                            ),
                        )
                        for i in drop:
                            mask[i] = 1
            # Case 3: no mask.
            elif self.params["masking"] == "none":
                # No mask at all.
                mask = [1] * len(self.vocab_act.keys())
            else:
                assert False, "Unrecognized masking {}".format(self.params["masking"])
            # print('mask ===>', mask)
            mask_all.append(mask)
        return torch.BoolTensor(mask_all).cuda().detach()

    def discount_reward(self, transitions, last_values):
        returns, advantages = [], []
        R = last_values.data
        for t in reversed(range(len(transitions))):
            _, _, values, rewards, done_masks, _, _, _, _, _, _ = transitions[t]
            R = rewards + self.params["gamma"] * R * done_masks
            adv = R - values
            returns.append(R)
            advantages.append(adv)
        return returns[::-1], advantages[::-1]

    def goexplore_train(
        self, obs, infos, graph_infos, max_steps, INTRINSIC_MOTIVTATION
    ):
        start = time.time()
        transitions = []
        if obs == None:
            obs, infos, graph_infos = self.vec_env.go_reset()
        for step in range(1, max_steps + 1):
            self.total_steps += 1
            tb.logkv("Step", self.total_steps)
            obs_reps = np.array([g.ob_rep for g in graph_infos])
            graph_mask_tt = self.generate_graph_mask(graph_infos)
            graph_state_reps = [g.graph_state_rep for g in graph_infos]

            if self.params["reward_type"] == "game_only":
                scores = [info["score"] for info in infos]
            elif self.params["reward_type"] == "IM_only":
                scores = np.array(
                    [
                        int(
                            len(INTRINSIC_MOTIVTATION[i])
                            * self.params["intrinsic_motivation_factor"]
                        )
                        for i in range(self.params["batch_size"])
                    ]
                )
            elif self.params["reward_type"] == "game_and_IM":
                scores = np.array(
                    [
                        infos[i]["score"]
                        + (
                            len(INTRINSIC_MOTIVTATION[i])
                            * (
                                (infos[i]["score"] + self.params["epsilon"])
                                / self.max_game_score
                            )
                        )
                        for i in range(self.params["batch_size"])
                    ]
                )

            (
                tmpl_pred_tt,
                obj_pred_tt,
                dec_obj_tt,
                dec_tmpl_tt,
                value,
                dec_steps,
                output_gat,
                query_important,
            ) = self.model(obs_reps, scores, graph_state_reps, graph_mask_tt)
            tb.logkv_mean("Value", value.mean().item())

            # Log some of the predictions and ground truth values
            topk_tmpl_probs, topk_tmpl_idxs = F.softmax(tmpl_pred_tt[0]).topk(5)
            topk_tmpls = [
                self.template_generator.templates[t] for t in topk_tmpl_idxs.tolist()
            ]
            tmpl_pred_str = ", ".join(
                [
                    "{} {:.3f}".format(tmpl, prob)
                    for tmpl, prob in zip(topk_tmpls, topk_tmpl_probs.tolist())
                ]
            )

            admissible = [g.admissible_actions for g in graph_infos]
            # print('admissible', admissible)
            objs = [g.objs for g in graph_infos]
            tmpl_gt_tt, obj_mask_gt_tt = self.generate_targets(admissible, objs)

            gt_tmpls = [
                self.template_generator.templates[i]
                for i in tmpl_gt_tt[0]
                .nonzero()
                .squeeze()
                .cpu()
                .numpy()
                .flatten()
                .tolist()
            ]
            gt_objs = [
                self.vocab_act[i]
                for i in obj_mask_gt_tt[0, 0]
                .nonzero()
                .squeeze()
                .cpu()
                .numpy()
                .flatten()
                .tolist()
            ]
            log("TmplPred: {} GT: {}".format(tmpl_pred_str, ", ".join(gt_tmpls)))
            topk_o1_probs, topk_o1_idxs = F.softmax(obj_pred_tt[0, 0]).topk(5)
            topk_o1 = [self.vocab_act[o] for o in topk_o1_idxs.tolist()]
            o1_pred_str = ", ".join(
                [
                    "{} {:.3f}".format(o, prob)
                    for o, prob in zip(topk_o1, topk_o1_probs.tolist())
                ]
            )
            graph_mask_str = [
                self.vocab_act[i]
                for i in graph_mask_tt[0]
                .nonzero()
                .squeeze()
                .cpu()
                .numpy()
                .flatten()
                .tolist()
            ]
            log(
                "ObjtPred: {} GT: {} Mask: {}".format(
                    o1_pred_str, ", ".join(gt_objs), ", ".join(graph_mask_str)
                )
            )

            chosen_actions = self.decode_actions(dec_tmpl_tt, dec_obj_tt)

            # Chooses random valid-actions to execute

            obs, rewards, dones, infos, graph_infos = self.vec_env.go_step(
                chosen_actions
            )
            # print('obs =>', obs)

            edges = [set(graph_info.graph_state.edges) for graph_info in graph_infos]
            size_updates = [0] * self.params["batch_size"]
            for i, s in enumerate(INTRINSIC_MOTIVTATION):
                orig_size = len(s)
                s.update(edges[i])
                size_updates[i] = len(s) - orig_size
            rewards = list(rewards)
            for i in range(self.params["batch_size"]):
                if self.params["reward_type"] == "IM_only":
                    rewards[i] = (
                        size_updates[i] * self.params["intrinsic_motivation_factor"]
                    )
                elif self.params["reward_type"] == "game_and_IM":
                    rewards[i] += (
                        size_updates[i] * self.params["intrinsic_motivation_factor"]
                    )
            rewards = tuple(rewards)

            tb.logkv_mean(
                "TotalStepsPerEpisode",
                sum([i["steps"] for i in infos]) / float(len(graph_infos)),
            )
            tb.logkv_mean("Valid", infos[0]["valid"])
            log(
                "Act: {}, Rew {}, Score {}, Done {}, Value {:.3f}".format(
                    chosen_actions[0],
                    rewards[0],
                    infos[0]["score"],
                    dones[0],
                    value[0].item(),
                )
            )
            log("Obs: {}".format(clean(obs[0])))
            if dones[0]:
                log("Step {} EpisodeScore {}\n".format(step, infos[0]["score"]))
            for done, info in zip(dones, infos):
                if done:
                    tb.logkv_mean("EpisodeScore", info["score"])
            rew_tt = torch.FloatTensor(rewards).cuda().unsqueeze(1)

            done_mask_tt = (~torch.tensor(dones)).float().cuda().unsqueeze(1)
            self.model.reset_hidden(done_mask_tt)
            transitions.append(
                (
                    tmpl_pred_tt,
                    obj_pred_tt,
                    value,
                    rew_tt,
                    done_mask_tt,
                    tmpl_gt_tt,
                    dec_tmpl_tt,
                    dec_obj_tt,
                    obj_mask_gt_tt,
                    graph_mask_tt,
                    dec_steps,
                )
            )

            if len(transitions) >= self.params["bptt"]:
                tb.logkv("StepsPerSecond", float(step) / (time.time() - start))
                self.model.clone_hidden()
                obs_reps = np.array([g.ob_rep for g in graph_infos])
                graph_mask_tt = self.generate_graph_mask(graph_infos)
                graph_state_reps = [g.graph_state_rep for g in graph_infos]
                # scores = [info['score'] for info in infos]
                if self.params["reward_type"] == "game_only":
                    scores = [info["score"] for info in infos]
                elif self.params["reward_type"] == "IM_only":
                    scores = np.array(
                        [
                            int(
                                len(INTRINSIC_MOTIVTATION[i])
                                * self.params["intrinsic_motivation_factor"]
                            )
                            for i in range(self.params["batch_size"])
                        ]
                    )
                elif self.params["reward_type"] == "game_and_IM":
                    print('scores =>', infos[i]["score"])
                    scores = np.array(
                        [
                            infos[i]["score"]
                            + (
                                len(INTRINSIC_MOTIVTATION[i])
                                * (
                                    (infos[i]["score"] + self.params["epsilon"])
                                    / self.max_game_score
                                )
                            )
                            for i in range(self.params["batch_size"])
                        ]
                    )
                _, _, _, _, next_value, _, output_gat, query_important = self.model(
                    obs_reps, scores, graph_state_reps, graph_mask_tt
                )
                returns, advantages = self.discount_reward(transitions, next_value)
                log(
                    "Returns: ",
                    ", ".join(["{:.3f}".format(a[0].item()) for a in returns]),
                )
                log(
                    "Advants: ",
                    ", ".join(["{:.3f}".format(a[0].item()) for a in advantages]),
                )
                tb.logkv_mean("Advantage", advantages[-1].median().item())
                loss = self.update(transitions, returns, advantages)
                del transitions[:]
                self.model.restore_hidden()

            if step % self.params["checkpoint_interval"] == 0:
                parameters = {"model": self.model}
                torch.save(
                    parameters, os.path.join(self.params["output_dir"], "qbert" + str(self.seed) + ".pt")
                )

        # self.vec_env.close_extras()
        return (
            obs,
            rewards,
            dones,
            infos,
            graph_infos,
            scores,
            chosen_actions,
            INTRINSIC_MOTIVTATION,
        )

    def train(self, max_steps):
        file_kg = open('/Q-BERT/qbert/logs/train_alpha_' +str(self.params['alpha_gat']) +'_' + game +'.txt', "w")
        print("=== === === start training!!! === === ===")
        start = time.time()
        if self.params["training_type"] == "chained":
            self.log_file(
                "BEGINNING OF TRAINING: patience={}, max_n_steps_back={}\n".format(
                    self.params["patience"], self.params["buffer_size"]
                )
            )
        frozen_policies = []
        transitions = []
        self.back_step = -1

        previous_best_seen_score = float("-inf")
        previous_best_step = 0
        previous_best_state = None
        previous_best_snapshot = None
        previous_best_ACTUAL_score = 0
        self.cur_reload_step = 0
        force_reload = [False] * self.params["batch_size"]
        last_edges = None

        self.valid_track = np.zeros(self.params["batch_size"])
        self.stagnant_steps = 0

        INTRINSIC_MOTIVTATION = [set() for i in range(self.params["batch_size"])]

        obs, infos, graph_infos, env_str = self.vec_env.reset()
        snap_obs = obs[0]
        snap_info = infos[0]
        snap_graph_reps = None
        # print (obs)
        # print (infos)
        # print (graph_infos)

        # early stop counting#
        best_scores = [0] * self.batch_size  # 8 env
        for step in range(1, max_steps + 1):
            # Step 1: build model inputs
            wallclock = time.time()

            if any(force_reload) and self.params["training_type"] == "chained":
                num_reload = force_reload.count(True)
                t_obs = np.array(obs)
                t_obs[force_reload] = [snap_obs] * num_reload
                obs = tuple(t_obs)

                t_infos = np.array(infos)
                t_infos[force_reload] = [snap_info] * num_reload
                infos = tuple(t_infos)

                t_graphs = list(graph_infos)
                # namedtuple gets lost in np.array
                t_updates = self.vec_env.load_from(
                    self.cur_reload_state, force_reload, snap_graph_reps, snap_obs
                )
                for i in range(self.params["batch_size"]):
                    if force_reload[i]:
                        t_graphs[i] = t_updates[i]
                graph_infos = tuple(t_graphs)

                force_reload = [False] * self.params["batch_size"]

            tb.logkv("Step", step)
            obs_reps = np.array([g.ob_rep for g in graph_infos])
            graph_mask_tt = self.generate_graph_mask(graph_infos)
            graph_state_reps = [g.graph_state_rep for g in graph_infos]

            if self.params["reward_type"] == "game_only":
                scores = [info["score"] for info in infos]
            elif self.params["reward_type"] == "IM_only":
                scores = np.array(
                    [
                        int(
                            len(INTRINSIC_MOTIVTATION[i])
                            * self.params["intrinsic_motivation_factor"]
                        )
                        for i in range(self.params["batch_size"])
                    ]
                )
            elif self.params["reward_type"] == "game_and_IM":
                scores = np.array(
                    [
                        infos[i]["score"]
                        + (
                            len(INTRINSIC_MOTIVTATION[i])
                            * (
                                (infos[i]["score"] + self.params["epsilon"])
                                / self.max_game_score
                            )
                        )
                        for i in range(self.params["batch_size"])
                    ]
                )
            # print('!!!score', scores)
            # print('graph_infos ===>', graph_infos)
            graph_rep_1 = [g.graph_state_rep_1 for g in graph_infos]
            graph_rep_2 = [g.graph_state_rep_2 for g in graph_infos]
            graph_rep_3 = [g.graph_state_rep_3 for g in graph_infos]
            graph_rep_4 = [g.graph_state_rep_4 for g in graph_infos]
            graph_rep = [g.graph_state_rep for g in graph_infos]
            adj0 = torch.IntTensor(graph_rep[0][1]).cuda()
            # print('graph_shape=>', len(torch.nonzero(adj0 > 0)))

            # save logs

            # print('state_gat', self.model.state_gat.state_ent_emb.weight.sum(dim=-1).sum())
            file_kg.write('\n')
            file_kg.write('=' * 30 + '\n')
            file_kg.write('STEP: ' + str(step) + '\n')
            file_kg.write(obs + '\n')
            # file_kg.write('BEST >' + str(best_scores) + '\n')
            # file_kg.write('BEST w/o KG >' + str([infos[i]["score"] for i in range(len(infos))]) + '\n')
            # file_kg.write('Embedding sum: ' + str(float(self.model.state_gat.state_ent_emb.weight.sum(dim=-1).sum().cpu().detach().numpy())) + '\n')
            # file_kg.write('>>>Full-graph: \n')
            # file_kg.write('Length:' + str(len([g.graph_state.edges for g in graph_infos][0])) + '\n')
            # file_kg.write(str([g.graph_state.edges for g in graph_infos][0]) + '\n')
            # file_kg.write('>>>Subgraph_1: \n')
            # file_kg.write('Length: ' + str(len([g.graph_state_1.edges for g in graph_infos][0])) + '\n')
            # file_kg.write(str([g.graph_state_1.edges for g in graph_infos][0]) + '\n')
            #
            # file_kg.write('>>>Subgraph_2: \n')
            # file_kg.write('Length: ' + str(len([g.graph_state_2.edges for g in graph_infos][0])) + '\n')
            # file_kg.write(str([g.graph_state_2.edges for g in graph_infos][0]) + '\n')
            # file_kg.write('>>>Subgraph_3: \n')
            # file_kg.write('Length: ' + str(len([g.graph_state_3.edges for g in graph_infos][0])) + '\n')
            # file_kg.write(str([g.graph_state_3.edges for g in graph_infos][0]) + '\n')
            # file_kg.write('>>>Subgraph_4: \n')
            # file_kg.write('Length: ' + str(len([g.graph_state_4.edges for g in graph_infos][0])) + '\n')
            # file_kg.write(str([g.graph_state_4.edges for g in graph_infos][0]) + '\n')
            file_kg.flush()
            # Step 2: predict probs, actual items
            (
                tmpl_pred_tt,
                obj_pred_tt,
                dec_obj_tt,
                dec_tmpl_tt,
                value,
                dec_steps,
                output_gat,
                query_important,
            ) = self.model(
                obs_reps,
                scores,
                graph_state_reps,
                graph_rep_1,
                graph_rep_2,
                graph_rep_3,
                graph_rep_4,
                graph_mask_tt,
            )

            # tmpl_pred_tt, obj_pred_tt, dec_obj_tt, dec_tmpl_tt, value, dec_steps = self.model(
            #     obs_reps, scores, graph_state_reps, graph_mask_tt)
            tb.logkv_mean("Value", value.mean().item())

            # Step 3: Log the predictions and ground truth values
            # Log the predictions and ground truth values
            topk_tmpl_probs, topk_tmpl_idxs = F.softmax(tmpl_pred_tt[0]).topk(5)
            topk_tmpls = [
                self.template_generator.templates[t] for t in topk_tmpl_idxs.tolist()
            ]
            tmpl_pred_str = ", ".join(
                [
                    "{} {:.3f}".format(tmpl, prob)
                    for tmpl, prob in zip(topk_tmpls, topk_tmpl_probs.tolist())
                ]
            )

            # Step 4: Generate the ground truth and object mask
            admissible = [g.admissible_actions for g in graph_infos]
            objs = [g.objs for g in graph_infos]
            tmpl_gt_tt, obj_mask_gt_tt = self.generate_targets(admissible, objs)

            # Step 5 Log template/object predictions/ground_truth
            gt_tmpls = [
                self.template_generator.templates[i]
                for i in tmpl_gt_tt[0]
                .nonzero()
                .squeeze()
                .cpu()
                .numpy()
                .flatten()
                .tolist()
            ]
            gt_objs = [
                self.vocab_act[i]
                for i in obj_mask_gt_tt[0, 0]
                .nonzero()
                .squeeze()
                .cpu()
                .numpy()
                .flatten()
                .tolist()
            ]
            log("TmplPred: {} GT: {}".format(tmpl_pred_str, ", ".join(gt_tmpls)))
            topk_o1_probs, topk_o1_idxs = F.softmax(obj_pred_tt[0, 0]).topk(5)
            topk_o1 = [self.vocab_act[o] for o in topk_o1_idxs.tolist()]
            o1_pred_str = ", ".join(
                [
                    "{} {:.3f}".format(o, prob)
                    for o, prob in zip(topk_o1, topk_o1_probs.tolist())
                ]
            )
            # graph_mask_str = [self.vocab_act[i] for i in graph_mask_tt[0].nonzero().squeeze().cpu().numpy().flatten().tolist()]
            log(
                "ObjtPred: {} GT: {}".format(o1_pred_str, ", ".join(gt_objs))
            )  # , ', '.join(graph_mask_str)))

            chosen_actions = self.decode_actions(dec_tmpl_tt, dec_obj_tt)

            # stepclock = time.time()
            # Step 6: Next step
            obs, rewards, dones, infos, graph_infos, env_str = self.vec_env.step(
                chosen_actions
            )
            # print('rewards =>', rewards)
            # print('stepclock', time.time() - stepclock)
            self.valid_track += [info["valid"] for info in infos]
            self.stagnant_steps += 1
            force_reload = list(dones)

            edges = [set(graph_info.graph_state.edges) for graph_info in graph_infos]
            size_updates = [0] * self.params["batch_size"]
            for i, s in enumerate(INTRINSIC_MOTIVTATION):
                orig_size = len(s)
                s.update(edges[i])
                size_updates[i] = len(s) - orig_size
            rewards = list(rewards)
            for i in range(self.params["batch_size"]):
                if self.params["reward_type"] == "IM_only":
                    rewards[i] = (
                        size_updates[i] * self.params["intrinsic_motivation_factor"]
                    )
                elif self.params["reward_type"] == "game_and_IM":
                    rewards[i] += (
                        size_updates[i] * self.params["intrinsic_motivation_factor"]
                    )
            rewards = tuple(rewards)

            if last_edges:
                stayed_same = [
                    1
                    if (
                        len(edges[i] - last_edges[i])
                        <= self.params["kg_diff_threshold"]
                    )
                    else 0
                    for i in range(self.params["batch_size"])
                ]
                # print ("stayed_same: {}".format(stayed_same))
            valid_kg_update = (
                last_edges
                and sum(stayed_same) / self.params["batch_size"]
                > self.params["kg_diff_batch_percentage"]
            )
            last_edges = edges

            snapshot = self.vec_env.get_snapshot()
            real_scores = np.array([infos[i]["score"] for i in range(len(rewards))])

            if self.params["reward_type"] == "game_only":
                scores = [info["score"] for info in infos]
            elif self.params["reward_type"] == "IM_only":
                scores = np.array(
                    [
                        int(
                            len(INTRINSIC_MOTIVTATION[i])
                            * self.params["intrinsic_motivation_factor"]
                        )
                        for i in range(self.params["batch_size"])
                    ]
                )
            elif self.params["reward_type"] == "game_and_IM":
                scores = np.array(
                    [
                        infos[i]["score"]
                        + (
                            len(INTRINSIC_MOTIVTATION[i])
                            * (
                                (infos[i]["score"] + self.params["epsilon"])
                                / self.max_game_score
                            )
                        )
                        for i in range(self.params["batch_size"])
                    ]
                )
            cur_max_score_idx = np.argmax(scores)
            if (
                scores[cur_max_score_idx] > previous_best_seen_score
                and self.params["training_type"] == "chained"
            ):  # or valid_kg_update:
                print("New Reward Founded OR KG updated")
                previous_best_step = step
                previous_best_state = env_str[cur_max_score_idx]
                previous_best_seen_score = scores[cur_max_score_idx]
                previous_best_snapshot = snapshot[cur_max_score_idx]
                self.back_step = -1
                self.valid_track = np.zeros(self.params["batch_size"])
                self.stagnant_steps = 0
                print("\tepoch: {}".format(previous_best_step))
                print("\tnew score: {}".format(previous_best_seen_score))
                print("\tthis info: {}".format(infos[cur_max_score_idx]))
                self.log_file(
                    "New High Score Founded: step:{}, new_score:{}, infos:{}\n".format(
                        previous_best_step,
                        previous_best_seen_score,
                        infos[cur_max_score_idx],
                    )
                )

            previous_best_ACTUAL_score = max(
                np.max(real_scores), previous_best_ACTUAL_score
            )
            print(
                "step {}: scores: {}, best_real_score: {}".format(
                    step, scores, previous_best_ACTUAL_score
                )
            )

            tb.logkv_mean(
                "TotalStepsPerEpisode",
                sum([i["steps"] for i in infos]) / float(len(graph_infos)),
            )
            tb.logkv_mean("Valid", infos[0]["valid"])
            log(
                "Act: {}, Rew {}, Score {}, Done {}, Value {:.3f}".format(
                    chosen_actions[0],
                    rewards[0],
                    infos[0]["score"],
                    dones[0],
                    value[0].item(),
                )
            )
            log("Obs: {}".format(clean(obs[0])))
            if dones[0]:
                log("Step {} EpisodeScore {}\n".format(step, infos[0]["score"]))
            for done, info in zip(dones, infos):
                if done:
                    tb.logkv_mean("EpisodeScore", info["score"])

            # Step 8: append into transitions
            rew_tt = torch.FloatTensor(rewards).cuda().unsqueeze(1)
            done_mask_tt = (~torch.tensor(dones)).float().cuda().unsqueeze(1)
            self.model.reset_hidden(done_mask_tt)
            transitions.append(
                (
                    tmpl_pred_tt,
                    obj_pred_tt,
                    value,
                    rew_tt,
                    done_mask_tt,
                    tmpl_gt_tt,
                    dec_tmpl_tt,
                    dec_obj_tt,
                    obj_mask_gt_tt,
                    graph_mask_tt,
                    dec_steps,
                )
            )

            # Step 9: update model per 8 steps
            if len(transitions) >= self.params["bptt"]:
                tb.logkv("StepsPerSecond", float(step) / (time.time() - start))
                self.model.clone_hidden()
                obs_reps = np.array([g.ob_rep for g in graph_infos])
                graph_mask_tt = self.generate_graph_mask(graph_infos)
                graph_state_reps = [g.graph_state_rep for g in graph_infos]
                graph_rep_1 = [g.graph_state_rep_1 for g in graph_infos]
                graph_rep_2 = [g.graph_state_rep_2 for g in graph_infos]
                graph_rep_3 = [g.graph_state_rep_3 for g in graph_infos]
                graph_rep_4 = [g.graph_state_rep_4 for g in graph_infos]

                if self.params["reward_type"] == "game_only":
                    scores = [info["score"] for info in infos]
                elif self.params["reward_type"] == "IM_only":
                    scores = np.array(
                        [
                            int(
                                len(INTRINSIC_MOTIVTATION[i])
                                * self.params["intrinsic_motivation_factor"]
                            )
                            for i in range(self.params["batch_size"])
                        ]
                    )
                elif self.params["reward_type"] == "game_and_IM":
                    scores = np.array(
                        [
                            infos[i]["score"]
                            + (
                                len(INTRINSIC_MOTIVTATION[i])
                                * (
                                    (infos[i]["score"] + self.params["epsilon"])
                                    / self.max_game_score
                                )
                            )
                            for i in range(self.params["batch_size"])
                        ]
                    )

                _, _, _, _, next_value, _, output_gat, query_important = self.model(
                    obs_reps,
                    scores,
                    graph_state_reps,
                    graph_rep_1,
                    graph_rep_2,
                    graph_rep_3,
                    graph_rep_4,
                    graph_mask_tt,
                )

                returns, advantages = self.discount_reward(transitions, next_value)
                log(
                    "Returns: ",
                    ", ".join(["{:.3f}".format(a[0].item()) for a in returns]),
                )
                log(
                    "Advants: ",
                    ", ".join(["{:.3f}".format(a[0].item()) for a in advantages]),
                )
                tb.logkv_mean("Advantage", advantages[-1].median().item())
                loss = self.update(transitions, returns, advantages)
                print("next_value ===>", next_value)
                del transitions[:]
                self.model.restore_hidden()

            if step % self.params["checkpoint_interval"] == 0:
                parameters = {"model": self.model}
                torch.save(
                    parameters, os.path.join(self.params["output_dir"], "qbert" + str(self.seed) + ".pt")
                )

            #########EARLY STOP#############
            early_stop_count = 0
            print("best_scores history =>", best_scores)
            for idx, score in enumerate(scores):
                best_scores[idx] = max(score, best_scores[idx])
                if best_scores[idx] < self.early_stop_score:
                    early_stop_count += 1
            print("early_stop counts are => ", early_stop_count)
            if early_stop_count <= self.early_stop_score_count:
                parameters = {"model": self.model}
                torch.save(
                    parameters, os.path.join(self.params["output_dir"], "qbert" + str(self.seed) + ".pt")
                )
                break
            #########EARLY STOP#############

            bottleneck = self.params["training_type"] == "chained" and (
                (
                    self.stagnant_steps >= self.params["patience"]
                    and not self.params["patience_valid_only"]
                )
                or (
                    self.params["patience_valid_only"]
                    and sum(self.valid_track >= self.params["patience"])
                    >= self.params["batch_size"] * self.params["patience_batch_factor"]
                )
            )
            if bottleneck:
                bottleneck_sig = True
                print("Bottleneck detected at step: {}".format(step))
                # new_backstep += 1
                # new_back_step = (step - previous_best_step - self.params['patience']) // self.params['patience']
                self.back_step += 1
                if self.back_step == 0:
                    self.vec_env.import_snapshot(previous_best_snapshot)
                    cur_time = time.strftime("%Y%m%d-%H%M%S")
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(self.chkpt_path, "{}.pt".format(cur_time)),
                    )
                    frozen_policies.append((cur_time, previous_best_state))
                    # INTRINSIC_MOTIVTATION= [set() for i in range(self.params['batch_size'])]
                    self.log_file(
                        "Current model saved at: model/checkpoints/{}.pt\n".format(
                            cur_time
                        )
                    )
                self.model = QBERT(
                    self.params,
                    self.template_generator.templates,
                    self.max_word_length,
                    self.vocab_act,
                    self.vocab_act_rev,
                    len(self.sp),
                    gat=self.params["gat"],
                    argmax_sig=self.params["argmax_sig"]
                ).cuda()

                if self.back_step >= self.params["buffer_size"]:
                    print("Buffer exhausted. Finishing training")
                    self.vec_env.close_extras()
                    return

                print(previous_best_snapshot[-1 - self.back_step])
                (
                    snap_obs,
                    snap_info,
                    snap_graph_reps,
                    self.cur_reload_state,
                ) = previous_best_snapshot[-1 - self.back_step]
                snap_obs, snap_info, snap_graph_reps, self.cur_reload_state = previous_best_snapshot[
                    -1 - self.back_step]
                print("Loading snapshot, infos: {}".format(snap_info))
                self.log_file("Loading snapshot, infos: {}\n".format(snap_info))
                self.cur_reload_step = previous_best_step
                force_reload = [True] * self.params["batch_size"]
                self.valid_track = np.zeros(self.params["batch_size"])
                self.stagnant_steps = 0

                # print out observations here
                # print(
                #     "Current observations: {}".format([info["look"] for info in infos])
                # )
                # print(
                #     "Previous_best_step: {}, step_back: {}".format(
                #         previous_best_step, self.back_step
                #     )
                # )
                # self.log_file(
                #     "Bottleneck Detected: step:{}, previous_best_step:{}, cur_step_back:{}\n".format(
                #         i, previous_best_step, self.back_step
                #     )
                # )
                # self.log_file(
                #     "Current observations: {}\n".format(
                #         [info["look"] for info in infos]
                #     )
                # )
            # exit()

        self.vec_env.close_extras()

    def update(self, transitions, returns, advantages):
        assert len(transitions) == len(returns) == len(advantages)
        loss = 0
        for trans, ret, adv in zip(transitions, returns, advantages):
            (
                tmpl_pred_tt,
                obj_pred_tt,
                value,
                _,
                _,
                tmpl_gt_tt,
                dec_tmpl_tt,
                dec_obj_tt,
                obj_mask_gt_tt,
                graph_mask_tt,
                dec_steps,
            ) = trans

            # Supervised Template Loss
            tmpl_probs = F.softmax(tmpl_pred_tt, dim=1)
            template_loss = self.params["template_coeff"] * self.loss_fn1(
                tmpl_probs, tmpl_gt_tt
            )

            # Supervised Object Loss
            if self.params["batch_size"] == 1:
                object_mask_target = obj_mask_gt_tt.unsqueeze(0).permute((1, 0, 2))
            else:
                object_mask_target = obj_mask_gt_tt.permute((1, 0, 2))
            obj_probs = F.softmax(obj_pred_tt, dim=2)
            object_mask_loss = self.params["object_coeff"] * self.loss_fn1(
                obj_probs, object_mask_target
            )

            # Build the object mask
            o1_mask, o2_mask = [0] * self.batch_size, [0] * self.batch_size
            for d, st in enumerate(dec_steps):
                if st > 1:
                    o1_mask[d] = 1
                    o2_mask[d] = 1
                elif st == 1:
                    o1_mask[d] = 1
            o1_mask = torch.FloatTensor(o1_mask).cuda()
            o2_mask = torch.FloatTensor(o2_mask).cuda()

            # Policy Gradient Loss
            policy_obj_loss = torch.FloatTensor([0]).cuda()
            cnt = 0
            for i in range(self.batch_size):
                if dec_steps[i] >= 1:
                    cnt += 1
                    batch_pred = obj_pred_tt[0, i, graph_mask_tt[i]]
                    action_log_probs_obj = F.log_softmax(batch_pred, dim=0)
                    dec_obj_idx = dec_obj_tt[0, i].item()
                    graph_mask_list = (
                        graph_mask_tt[i]
                        .nonzero()
                        .squeeze()
                        .cpu()
                        .numpy()
                        .flatten()
                        .tolist()
                    )
                    idx = graph_mask_list.index(dec_obj_idx)
                    log_prob_obj = action_log_probs_obj[idx]
                    policy_obj_loss += -log_prob_obj * adv[i].detach()
            if cnt > 0:
                policy_obj_loss /= cnt
            tb.logkv_mean("PolicyObjLoss", policy_obj_loss.item())
            log_probs_obj = F.log_softmax(obj_pred_tt, dim=2)

            log_probs_tmpl = F.log_softmax(tmpl_pred_tt, dim=1)
            action_log_probs_tmpl = log_probs_tmpl.gather(1, dec_tmpl_tt).squeeze()

            policy_tmpl_loss = (-action_log_probs_tmpl * adv.detach().squeeze()).mean()
            tb.logkv_mean("PolicyTemplateLoss", policy_tmpl_loss.item())

            policy_loss = policy_tmpl_loss + policy_obj_loss

            value_loss = self.params["value_coeff"] * self.loss_fn3(value, ret)
            tmpl_entropy = -(tmpl_probs * log_probs_tmpl).mean()
            tb.logkv_mean("TemplateEntropy", tmpl_entropy.item())
            object_entropy = -(obj_probs * log_probs_obj).mean()
            tb.logkv_mean("ObjectEntropy", object_entropy.item())
            # Minimizing entropy loss will lead to increased entropy
            entropy_loss = self.params["entropy_coeff"] * -(
                tmpl_entropy + object_entropy
            )

            loss += (
                template_loss
                + object_mask_loss
                + value_loss
                + entropy_loss
                + policy_loss
            )

        tb.logkv("Loss", loss.item())
        tb.logkv("TemplateLoss", template_loss.item())
        tb.logkv("ObjectLoss", object_mask_loss.item())
        tb.logkv("PolicyLoss", policy_loss.item())
        tb.logkv("ValueLoss", value_loss.item())
        tb.logkv("EntropyLoss", entropy_loss.item())
        tb.dumpkvs()
        loss.backward()

        # Compute the gradient norm
        grad_norm = 0
        for p in list(filter(lambda p: p.grad is not None, self.model.parameters())):
            grad_norm += p.grad.data.norm(2).item()
        tb.logkv("UnclippedGradNorm", grad_norm)

        nn.utils.clip_grad_norm_(self.model.parameters(), self.params["clip"])

        # Clipped Grad norm
        grad_norm = 0
        for p in list(filter(lambda p: p.grad is not None, self.model.parameters())):
            grad_norm += p.grad.data.norm(2).item()
        tb.logkv("ClippedGradNorm", grad_norm)

        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss

    def decode_actions(self, decoded_templates, decoded_objects):
        """
        Returns string representations of the given template actions.

        :param decoded_template: Tensor of template indices.
        :type decoded_template: Torch tensor of size (Batch_size x 1).
        :param decoded_objects: Tensor of o1, o2 object indices.
        :type decoded_objects: Torch tensor of size (2 x Batch_size x 1).

        """
        decoded_actions = []
        for i in range(self.batch_size):
            decoded_template = decoded_templates[i].item()
            decoded_object1 = decoded_objects[0][i].item()
            decoded_object2 = decoded_objects[1][i].item()
            decoded_action = self.tmpl_to_str(
                decoded_template, decoded_object1, decoded_object2
            )
            decoded_actions.append(decoded_action)
        return decoded_actions

    def tmpl_to_str(self, template_idx, o1_id, o2_id):
        """ Returns a string representation of a template action. """
        template_str = self.template_generator.templates[template_idx]
        holes = template_str.count("OBJ")
        assert holes <= 2
        if holes <= 0:
            return template_str
        elif holes == 1:
            return template_str.replace("OBJ", self.vocab_act[o1_id])
        else:
            return template_str.replace("OBJ", self.vocab_act[o1_id], 1).replace(
                "OBJ", self.vocab_act[o2_id], 1
            )

    def explain_action(self, action, output_gat):
        """

        """
        print("=" * 20)
        print("Action is ", action)
        value_order = []
        rels = ["is", "have", "in", "others", "all"]
        for idx, (value, adj) in enumerate(output_gat):  # consider every subgraph
            for i in range(3):  # consider top 3
                value_node = int(value.indices[i].cpu().numpy())  # node's id
                node_value = value.values[i].cpu().detach().numpy()  # node's value
                if node_value > 0:
                    # print(int(adj.indices[0].cpu().numpy()))
                    for loc in range(adj.shape[0]):
                        # print(int(adj[loc][0].cpu().numpy()))
                        adj_node = int(adj[loc][0].cpu().numpy())
                        adj_node_obj = int(adj[loc][1].cpu().numpy())
                        if value_node == adj_node:
                            value_order.append(
                                [
                                    node_value,
                                    self.tsv_file[0][value_node]
                                    + " "
                                    + rels[idx]
                                    + " "
                                    + self.tsv_file[0][adj_node_obj],
                                ]
                            )
                            if i == 0:
                                print(
                                    idx,
                                    "graph",
                                    "!!!REASON =>",
                                    self.tsv_file[0][value_node],
                                    rels[idx],
                                    self.tsv_file[0][adj_node_obj],
                                )
        if value_order:
            value_order.sort(key=lambda x: x[0])
            print(value_order[-1][-1])
            if len(value_order) > 1:
                print(value_order[-2][-1])

        else:
            print("Reason cannot find in the subgraph!")
            print(output_gat)
        print("=" * 20)

    def load_vocab_kge(self, tsv_file):
        ent = {}
        with open(tsv_file, "r") as f:
            for line in f:
                e, eid = line.split("\t")
                ent[str(e.strip())] = int(eid.strip())
        ent_id = {v: k for k, v in ent.items()}
        rel_path = os.path.dirname(tsv_file)
        rel_name = os.path.join(rel_path, "relation2id.tsv")
        rel = {}
        with open(rel_name, "r") as f:
            for line in f:
                r, rid = line.split("\t")
                rel[int(rid.strip())] = r.strip()
        return ent_id, ent, rel

    def get_random_action(self, obs):
        valid_actions = self.vec_env.get_valid_actions(obs)
        # for valid_action in valid_actions:

        return valid_actions

    def test(self, max_steps):
        print("=== === === start Testing!!! === === ===")
        bottleneck = None
        bottleneck_sig = False  # when face bottleneck, it will be 1 until overcomed.
        # eval mode
        self.model.eval()
        #
        start = time.time()
        if self.params["training_type"] == "chained":
            self.log_file(
                "BEGINNING OF TRAINING: patience={}, max_n_steps_back={}\n".format(
                    self.params["patience"], self.params["buffer_size"]
                )
            )
        frozen_policies = []
        transitions = []
        self.back_step = -1

        previous_best_seen_score = float("-inf")
        previous_best_seen_score_noback = float("-inf")
        previous_best_step = 0
        previous_best_state = None
        previous_best_snapshot = None
        previous_best_ACTUAL_score = 0
        self.cur_reload_step = 0
        force_reload = [False] * self.params["batch_size"]
        last_edges = None

        self.valid_track = np.zeros(self.params["batch_size"])
        self.stagnant_steps = 0

        INTRINSIC_MOTIVTATION = [set() for i in range(self.params["batch_size"])]

        obs, infos, graph_infos, env_str = self.vec_env.reset()
        snap_obs = obs[0]
        snap_info = infos[0]
        snap_graph_reps = None
        prev_scores = [0] * 2
        reward_prev = False
        obs_before = "None"
        old_real_scores = 0

        for step in range(1, max_steps + 1):
            print("bottleneck_sig", bottleneck_sig)
            # Step 1: build model inputs
            if any(force_reload) and self.params["training_type"] == "chained":
                num_reload = force_reload.count(True)
                t_obs = np.array(obs)
                t_obs[force_reload] = [snap_obs] * num_reload
                obs = tuple(t_obs)

                t_infos = np.array(infos)
                t_infos[force_reload] = [snap_info] * num_reload
                infos = tuple(t_infos)

                t_graphs = list(graph_infos)
                # namedtuple gets lost in np.array
                t_updates = self.vec_env.load_from(
                    self.cur_reload_state, force_reload, snap_graph_reps, snap_obs
                )
                for i in range(self.params["batch_size"]):
                    if force_reload[i]:
                        t_graphs[i] = t_updates[i]
                graph_infos = tuple(t_graphs)

                force_reload = [False] * self.params["batch_size"]

            wallclock = time.time()
            tb.logkv("Step", step)
            obs_reps = np.array([g.ob_rep for g in graph_infos])
            graph_mask_tt = self.generate_graph_mask(graph_infos)
            graph_state_reps = [g.graph_state_rep for g in graph_infos]

            if self.params["reward_type"] == "game_only":
                scores = [info["score"] for info in infos]
            elif self.params["reward_type"] == "IM_only":
                scores = np.array(
                    [
                        int(
                            len(INTRINSIC_MOTIVTATION[i])
                            * self.params["intrinsic_motivation_factor"]
                        )
                        for i in range(self.params["batch_size"])
                    ]
                )
            elif self.params["reward_type"] == "game_and_IM":
                print('BBB', infos[0]["score"])
                scores = np.array(
                    [
                        infos[i]["score"]
                        + (
                            len(INTRINSIC_MOTIVTATION[i])
                            * (
                                (infos[i]["score"] + self.params["epsilon"])
                                / self.max_game_score
                            )
                        )
                        for i in range(self.params["batch_size"])
                    ]
                )
                print('AAA', scores)
            # print('graph_infos ===>', graph_infos)
            graph_rep = [g.graph_state_rep for g in graph_infos]
            graph_rep_1 = [g.graph_state_rep_1 for g in graph_infos]
            graph_rep_2 = [g.graph_state_rep_2 for g in graph_infos]
            graph_rep_3 = [g.graph_state_rep_3 for g in graph_infos]
            graph_rep_4 = [g.graph_state_rep_4 for g in graph_infos]
            # graph_state = [g.graph_state for g in graph_infos]

            # Step 2: predict probs, actual items
            (
                tmpl_pred_tt,
                obj_pred_tt,
                dec_obj_tt,
                dec_tmpl_tt,
                value,
                dec_steps,
                output_gat,
                query_important,
            ) = self.model(
                obs_reps,
                scores,
                graph_state_reps,
                graph_rep_1,
                graph_rep_2,
                graph_rep_3,
                graph_rep_4,
                graph_mask_tt,
            )

            # tmpl_pred_tt, obj_pred_tt, dec_obj_tt, dec_tmpl_tt, value, dec_steps = self.model(
            #     obs_reps, scores, graph_state_reps, graph_mask_tt)
            tb.logkv_mean("Value", value.mean().item())
            print('value', value[0].cpu().detach().numpy()[0])
            # print('query_important', query_important, self.sp.decode_ids(query_important))

            # Step 3: Log the predictions and ground truth values
            # Log the predictions and ground truth values
            topk_tmpl_probs, topk_tmpl_idxs = F.softmax(tmpl_pred_tt[0]).topk(5)
            topk_tmpls = [
                self.template_generator.templates[t] for t in topk_tmpl_idxs.tolist()
            ]
            tmpl_pred_str = ", ".join(
                [
                    "{} {:.3f}".format(tmpl, prob)
                    for tmpl, prob in zip(topk_tmpls, topk_tmpl_probs.tolist())
                ]
            )

            # Step 4: Generate the ground truth and object mask
            admissible = [g.admissible_actions for g in graph_infos]
            objs = [g.objs for g in graph_infos]
            tmpl_gt_tt, obj_mask_gt_tt = self.generate_targets(admissible, objs)

            # Step 5 Log template/object predictions/ground_truth
            gt_tmpls = [
                self.template_generator.templates[i]
                for i in tmpl_gt_tt[0]
                .nonzero()
                .squeeze()
                .cpu()
                .numpy()
                .flatten()
                .tolist()
            ]
            gt_objs = [
                self.vocab_act[i]
                for i in obj_mask_gt_tt[0, 0]
                .nonzero()
                .squeeze()
                .cpu()
                .numpy()
                .flatten()
                .tolist()
            ]
            log("TmplPred: {} GT: {}".format(tmpl_pred_str, ", ".join(gt_tmpls)))
            topk_o1_probs, topk_o1_idxs = F.softmax(obj_pred_tt[0, 0]).topk(5)
            topk_o1 = [self.vocab_act[o] for o in topk_o1_idxs.tolist()]
            o1_pred_str = ", ".join(
                [
                    "{} {:.3f}".format(o, prob)
                    for o, prob in zip(topk_o1, topk_o1_probs.tolist())
                ]
            )
            # graph_mask_str = [self.vocab_act[i] for i in graph_mask_tt[0].nonzero().squeeze().cpu().numpy().flatten().tolist()]
            log(
                "ObjtPred: {} GT: {}".format(o1_pred_str, ", ".join(gt_objs))
            )  # , ', '.join(graph_mask_str)))

            chosen_actions = self.decode_actions(dec_tmpl_tt, dec_obj_tt)
            # print('model =>', chosen_actions[0], dec_tmpl_tt, dec_obj_tt)
            if self.random_action:
                admissible = [g.admissible_actions for g in graph_infos]
                chosen_action_idx = random.randint(0, len(admissible[0]) - 1)
                chosen_action = admissible[0][chosen_action_idx].action
                chosen_actions = [chosen_action, chosen_action]
            else:
                chosen_actions = [chosen_actions[0], chosen_actions[0]]

            # Step 6: Next step

            # print adj
            adj1 = torch.IntTensor(graph_rep_1[0][1]).cuda()
            adj2 = torch.IntTensor(graph_rep_2[0][1]).cuda()
            adj3 = torch.IntTensor(graph_rep_3[0][1]).cuda()
            adj4 = torch.IntTensor(graph_rep_4[0][1]).cuda()
            adj0 = torch.IntTensor(graph_rep[0][1]).cuda()
            # print('000=>', self.XRL.adj_to_entity(torch.nonzero(adj0 > 0)))
            # print('111=>',  self.XRL.adj_to_entity(torch.nonzero(adj1 > 0)))
            # print('222=>',  self.XRL.adj_to_entity(torch.nonzero(adj2 > 0)))
            # print('333=>',  self.XRL.adj_to_entity(torch.nonzero(adj3 > 0)))
            # print('444=>',  self.XRL.adj_to_entity(torch.nonzero(adj4 > 0)))
            # print('=========================')
            graph_infos_prev = copy.deepcopy(graph_infos)
            obs, rewards, dones, infos, graph_infos, env_str = self.vec_env.step(
                chosen_actions
            )
            # print('rewards =>', rewards[0])

            print('VALID =>', [info["valid"] for info in infos])

            # if value[0].cpu().detach().numpy()[0] > 3:
            #     self.XRL.forward(
            #         chosen_actions[0],
            #         output_gat[0],
            #         rewards[0],
            #         obs_before,
            #         obs[0],
            #         graph_infos_prev,
            #         graph_infos,
            #         bottleneck=False,
            #         reward_change=True,
            #         value=value[0].cpu().detach().numpy()[0]
            #     )

            # if sum(rewards) != 0:
            #     for i, reward in enumerate(rewards):
            #         if i == 0 and reward != 0:
            #             print("The", i, "th env's reward is", reward)
            #             self.XRL.forward(
            #                 chosen_actions[i],
            #                 output_gat[i],
            #                 reward,
            #                 obs_before,
            #                 obs[0],
            #                 graph_infos_prev,
            #                 graph_infos,
            #                 bottleneck=False,
            #                 reward_change=True,
            #             )
            #             reward_prev = True

            reward_log = rewards[0]
            # update obs_before
            obs_before = obs[0] if obs[0].count(".") > 1 else obs_before
            # if rewards[0] > 0:  # and rewards[0] < 10:
            #     break
            print('!!!!!!!', obs_before)
            self.valid_track += [info["valid"] for info in infos]
            valid_sig = [info["valid"] for info in infos][0]
            self.stagnant_steps += 1
            force_reload = list(dones)

            edges = [set(graph_info.graph_state.edges) for graph_info in graph_infos]
            size_updates = [0] * self.params["batch_size"]
            for i, s in enumerate(INTRINSIC_MOTIVTATION):
                orig_size = len(s)
                s.update(edges[i])
                size_updates[i] = len(s) - orig_size
            rewards = list(rewards)
            for i in range(self.params["batch_size"]):
                if self.params["reward_type"] == "IM_only":
                    rewards[i] = (
                        size_updates[i] * self.params["intrinsic_motivation_factor"]
                    )
                elif self.params["reward_type"] == "game_and_IM":
                    rewards[i] += (
                        size_updates[i] * self.params["intrinsic_motivation_factor"]
                    )
            rewards = tuple(rewards)

            if last_edges:
                stayed_same = [
                    1
                    if (
                        len(edges[i] - last_edges[i])
                        <= self.params["kg_diff_threshold"]
                    )
                    else 0
                    for i in range(self.params["batch_size"])
                ]
                print ("stayed_same: {}".format(stayed_same))
            valid_kg_update = (
                last_edges
                and sum(stayed_same) / self.params["batch_size"]
                > self.params["kg_diff_batch_percentage"]
            )
            last_edges = edges

            snapshot = self.vec_env.get_snapshot()
            real_scores = np.array([infos[i]["score"] for i in range(len(rewards))])

            if self.params["reward_type"] == "game_only":
                scores = [info["score"] for info in infos]
            elif self.params["reward_type"] == "IM_only":
                scores = np.array(
                    [
                        int(
                            len(INTRINSIC_MOTIVTATION[i])
                            * self.params["intrinsic_motivation_factor"]
                        )
                        for i in range(self.params["batch_size"])
                    ]
                )
            elif self.params["reward_type"] == "game_and_IM":
                scores = np.array(
                    [
                        infos[i]["score"]
                        + (
                            len(INTRINSIC_MOTIVTATION[i])
                            * (
                                (infos[i]["score"] + self.params["epsilon"])
                                / self.max_game_score
                            )
                        )
                        for i in range(self.params["batch_size"])
                    ]
                )
            print('real_scores[0]', real_scores[0])
            if real_scores[0] > old_real_scores and valid_sig: # when we have new real score! and valid action
                self.XRL.forward(
                    step,
                    chosen_actions[0],
                    output_gat[0],
                    real_scores[0]-old_real_scores,
                    obs_before,
                    obs[0],
                    graph_infos_prev,
                    graph_infos,
                    bottleneck=False,
                    reward_change=True,
                    score=scores[0],
                )
                # reward_prev = True
                old_real_scores = real_scores[0]

            # log the info here!
            if valid_sig:
                self.XRL.log(
                    obs_reps=obs_reps,
                    action=chosen_actions[0],
                    bottleneck=bottleneck_sig,
                    step=step,
                    graph_info=graph_infos_prev,
                    reward=reward_log,
                    output_gat=output_gat[0],
                    query_important=query_important,
                    scores_after=scores,
                    obs_next=np.array([g.ob_rep for g in graph_infos])
                )

            cur_max_score_idx = np.argmax(scores)
            if (
                scores[cur_max_score_idx] > previous_best_seen_score
                and self.params["training_type"] == "chained"
            ):  # or valid_kg_update:
                print("New Reward Founded OR KG updated")
                if bottleneck_sig and valid_sig:  # bottle neck is solved, so we need to log the actions.
                    self.XRL.forward(
                        step,
                        chosen_actions[0],
                        output_gat[0],
                        reward_log,
                        obs_before,
                        obs[0],
                        graph_infos_bottleneck,
                        graph_infos,
                        bottleneck=False,
                        reward_change=False,
                        score=scores[cur_max_score_idx],
                    )
                    bottleneck_sig = False
                else:
                    if value[0].cpu().detach().numpy()[0] > 0.5 and valid_sig:
                        self.XRL.forward(
                            step,
                            chosen_actions[0],
                            output_gat[0],
                            reward_log,
                            obs_before,
                            obs[0],
                            graph_infos_prev,
                            graph_infos,
                            bottleneck=False,
                            reward_change=True,
                            value=value[0].cpu().detach().numpy()[0],
                            score=scores[cur_max_score_idx],
                        )

                previous_best_step = step
                previous_best_state = env_str[cur_max_score_idx]
                previous_best_seen_score = scores[cur_max_score_idx]
                previous_best_snapshot = snapshot[cur_max_score_idx]
                self.back_step = -1
                self.valid_track = np.zeros(self.params["batch_size"])
                self.stagnant_steps = 0
                print("\tepoch: {}".format(previous_best_step))
                print("\tnew score: {}".format(previous_best_seen_score))
                print("\tthis info: {}".format(infos[cur_max_score_idx]))
                self.log_file(
                    "New High Score Founded: step:{}, new_score:{}, infos:{}\n".format(
                        previous_best_step,
                        previous_best_seen_score,
                        infos[cur_max_score_idx],
                    )
                )

            previous_best_ACTUAL_score = max(
                np.max(real_scores), previous_best_ACTUAL_score
            )
            print(
                "step {}: scores: {}, best_real_score: {}".format(
                    step, scores, previous_best_ACTUAL_score
                )
            )
            # if prev_scores != scores:
            #     print(">>>>> step {}: scores: {}".format(step, scores))

            prev_scores = scores.copy()

            tb.logkv_mean(
                "TotalStepsPerEpisode",
                sum([i["steps"] for i in infos]) / float(len(graph_infos)),
            )
            tb.logkv_mean("Valid", infos[0]["valid"])
            log(
                "Act: {}, Rew {}, Score {}, Done {}, Value {:.3f}".format(
                    chosen_actions[0],
                    rewards[0],
                    infos[0]["score"],
                    dones[0],
                    value[0].item(),
                )
            )
            log("Obs: {}".format(clean(obs[0])))
            if dones[0]:
                log("Step {} EpisodeScore {}\n".format(step, infos[0]["score"]))
            for done, info in zip(dones, infos):
                if done:
                    tb.logkv_mean("EpisodeScore", info["score"])

            # Step 8: append into transitions
            rew_tt = torch.FloatTensor(rewards).cuda().unsqueeze(1)
            done_mask_tt = (~torch.tensor(dones)).float().cuda().unsqueeze(1)
            self.model.reset_hidden(done_mask_tt)
            transitions.append(
                (
                    tmpl_pred_tt,
                    obj_pred_tt,
                    value,
                    rew_tt,
                    done_mask_tt,
                    tmpl_gt_tt,
                    dec_tmpl_tt,
                    dec_obj_tt,
                    obj_mask_gt_tt,
                    graph_mask_tt,
                    dec_steps,
                )
            )

            # Step 9: update model per 8 steps
            if len(transitions) >= self.params["bptt"]:
                tb.logkv("StepsPerSecond", float(step) / (time.time() - start))
                self.model.clone_hidden()
                obs_reps = np.array([g.ob_rep for g in graph_infos])
                graph_mask_tt = self.generate_graph_mask(graph_infos)
                graph_state_reps = [g.graph_state_rep for g in graph_infos]
                graph_rep_1 = [g.graph_state_rep_1 for g in graph_infos]
                graph_rep_2 = [g.graph_state_rep_2 for g in graph_infos]
                graph_rep_3 = [g.graph_state_rep_3 for g in graph_infos]
                graph_rep_4 = [g.graph_state_rep_4 for g in graph_infos]

                if self.params["reward_type"] == "game_only":
                    scores = [info["score"] for info in infos]
                elif self.params["reward_type"] == "IM_only":
                    scores = np.array(
                        [
                            int(
                                len(INTRINSIC_MOTIVTATION[i])
                                * self.params["intrinsic_motivation_factor"]
                            )
                            for i in range(self.params["batch_size"])
                        ]
                    )
                elif self.params["reward_type"] == "game_and_IM":
                    scores = np.array(
                        [
                            infos[i]["score"]
                            + (
                                len(INTRINSIC_MOTIVTATION[i])
                                * (
                                    (infos[i]["score"] + self.params["epsilon"])
                                    / self.max_game_score
                                )
                            )
                            for i in range(self.params["batch_size"])
                        ]
                    )

                _, _, _, _, next_value, _, output_gat, query_important = self.model(
                    obs_reps,
                    scores,
                    graph_state_reps,
                    graph_rep_1,
                    graph_rep_2,
                    graph_rep_3,
                    graph_rep_4,
                    graph_mask_tt,
                )

                returns, advantages = self.discount_reward(transitions, next_value)
                log(
                    "Returns: ",
                    ", ".join(["{:.3f}".format(a[0].item()) for a in returns]),
                )
                log(
                    "Advants: ",
                    ", ".join(["{:.3f}".format(a[0].item()) for a in advantages]),
                )
                tb.logkv_mean("Advantage", advantages[-1].median().item())
                # loss = self.update(transitions, returns, advantages)
                # print('next_value ===>', next_value)
                del transitions[:]
                self.model.restore_hidden()

            bottleneck = self.params["training_type"] == "chained" and (
                (
                    self.stagnant_steps >= self.params["patience"]
                    and not self.params["patience_valid_only"]
                )
                or (
                    self.params["patience_valid_only"]
                    and sum(self.valid_track >= self.params["patience"])
                    >= self.params["batch_size"] * self.params["patience_batch_factor"]
                )
            )
            print('bottleneck =>', bottleneck)

            if bottleneck:
                print("Bottleneck detected at step: {}".format(step))
                graph_infos_bottleneck = copy.deepcopy(graph_infos_prev)
                # log the data
                if not bottleneck_sig:
                    self.XRL.forward(
                        step,
                        chosen_actions[0],
                        output_gat[0],
                        reward_log,
                        obs_before,
                        obs[0],
                        graph_infos_prev,
                        graph_infos,
                        bottleneck=True,
                        reward_change=False,
                    )
                bottleneck_sig = True
                # new_backstep += 1
                # new_back_step = (step - previous_best_step - self.params['patience']) // self.params['patience']
                self.back_step += 1

                if self.back_step == 0:
                    self.vec_env.import_snapshot(previous_best_snapshot)
                    cur_time = time.strftime("%Y%m%d-%H%M%S")
                    frozen_policies.append((cur_time, previous_best_state))





                self.cur_reload_step = previous_best_step
                # BBB -> stop reload
                force_reload = [True] * self.params["batch_size"]
                self.valid_track = np.zeros(self.params["batch_size"])
                self.stagnant_steps = 0

                # print out observations here
                print("Current observations: {}".format([info for info in infos]))
                print(
                    "Previous_best_step: {}, step_back: {}".format(
                        previous_best_step, self.back_step
                    )
                )
                self.log_file(
                    "Bottleneck Detected: step:{}, previous_best_step:{}, cur_step_back:{}\n".format(
                        i, previous_best_step, self.back_step
                    )
                )
                self.log_file(
                    "Current observations: {}\n".format([info for info in infos])
                )

        self.vec_env.close_extras()

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.autograd as autograd
# import torch.nn.functional as F
# import os
# from os.path import basename, splitext
# import numpy as np
# import time
# import sentencepiece as spm
# from statistics import mean
#
# from jericho import *
# from jericho.template_action_generator import TemplateActionGenerator
# from jericho.util import unabbreviate, clean
# import jericho.defines
#
# # from representations import StateAction
# from models import QBERT
# from env import *
# from vec_env import *
# import logger
#
# import random
#
# # from extraction import kgextraction
# # torch.backends.cudnn.deterministic = True
# # torch.backends.cudnn.benchmark = False
# device = torch.device("cuda")

#
# def configure_logger(log_dir):
#     logger.configure(log_dir, format_strs=['log'])
#     global tb
#     tb = logger.Logger(log_dir, [logger.make_output_format('tensorboard', log_dir),
#                                  logger.make_output_format('csv', log_dir),
#                                  logger.make_output_format('stdout', log_dir)])
#     global log
#     logger.set_level(60)
#     log = logger.log
#
#
# class QBERTTrainer(object):
#     '''
#
#     QBERT main class.
#
#
#     '''
#
#     def __init__(self, params, args):
#         print("----- Initiating ----- ")
#         print("----- step 1 configure logger")
#         torch.manual_seed(params['seed'])
#         np.random.seed(params['seed'])
#         random.seed(params['seed'])
#         configure_logger(params['output_dir'])
#         log('Parameters {}'.format(params))
#         self.params = params
#         self.chkpt_path = os.path.dirname(self.params['checkpoint_path'])
#         if not os.path.exists(self.chkpt_path):
#             os.mkdir(self.chkpt_path)
#         print("----- step 2 load pre-collected things")
#         self.binding = load_bindings(params['rom_file_path'])
#         self.max_word_length = self.binding['max_word_length']
#         self.sp = spm.SentencePieceProcessor()
#         self.sp.Load(params['spm_file'])
#         # askbert_args = {'input_text': '', 'length': 10, 'batch_size': 1, 'temperature': 1, 'model_name': '117M',
#         #                'seed': 0, 'nsamples': 10, 'cutoffs': "6.5 -7 -5", 'write_sfdp': False, 'random': False}
#         # self.extraction = kgextraction.World([], [], [], askbert_args)
#         self.askbert = params['extraction']
#         print("----- step 3 build QBERTEnv")
#         kg_env = QBERTEnv(rom_path=params['rom_file_path'],
#                           seed=params['seed'],
#                           spm_model=self.sp,
#                           tsv_file=params['tsv_file'],
#                           attr_file=params['attr_file'],
#                           step_limit=params['reset_steps'],
#                           stuck_steps=params['stuck_steps'],
#                           gat=params['gat'],
#                           askbert=self.askbert,
#                           clear_kg=params['clear_kg_on_reset'],
#                           subKG_type=params['subKG_type'])
#         self.vec_env = VecEnv(num_envs=params['batch_size'],
#                               env=kg_env,
#                               openie_path=params['openie_path'],
#                               redis_db_path=params['redis_db_path'],
#                               buffer_size=params['buffer_size'],
#                               askbert=params['extraction'],
#                               training_type=params['training_type'],
#                               clear_kg=params['clear_kg_on_reset'])
#         self.template_generator = TemplateActionGenerator(self.binding)
#         print("----- step 4 build FrotzEnv and templace generator")
#         env = FrotzEnv(params['rom_file_path'])
#         self.max_game_score = env.get_max_score()
#         self.cur_reload_state = env.get_state()
#         self.vocab_act, self.vocab_act_rev = load_vocab(env)
#         print("----- step 5 build Qbert model")
#         self.model = QBERT(params,
#                            self.template_generator.templates,
#                            self.max_word_length,
#                            self.vocab_act,
#                            self.vocab_act_rev,
#                            len(self.sp),
#                            gat=self.params['gat']).cuda()
#         print("----- step 6 set training parameters")
#         self.batch_size = params['batch_size']
#         if params['preload_weights']:
#             self.model = torch.load(self.params['preload_weights'])['model']
#         self.optimizer = optim.Adam(self.model.parameters(), lr=params['lr'])
#
#         self.loss_fn1 = nn.BCELoss()
#         self.loss_fn2 = nn.BCEWithLogitsLoss()
#         self.loss_fn3 = nn.MSELoss()
#
#         self.chained_logger = params['chained_logger']
#         self.total_steps = 0
#         print("----- Init finished! ----- ")
#
#     def log_file(self, str):
#         with open(self.chained_logger, 'a+') as fh:
#             fh.write(str)
#
#     def generate_targets(self, admissible, objs):
#         '''
#         Generates ground-truth targets for admissible actions.
#
#         :param admissible: List-of-lists of admissible actions. Batch_size x Admissible
#         :param objs: List-of-lists of interactive objects. Batch_size x Objs
#         :returns: template targets and object target tensors
#
#         '''
#         tmpl_target = []
#         obj_targets = []
#         for adm in admissible:
#             obj_t = set()
#             cur_t = [0] * len(self.template_generator.templates)
#             for a in adm:
#                 cur_t[a.template_id] = 1
#                 obj_t.update(a.obj_ids)
#             tmpl_target.append(cur_t)
#             obj_targets.append(list(obj_t))
#         tmpl_target_tt = torch.FloatTensor(tmpl_target).cuda()
#
#         # Note: Adjusted to use the objects in the admissible actions only
#         object_mask_target = []
#         for objl in obj_targets:  # in objs
#             cur_objt = [0] * len(self.vocab_act)
#             for o in objl:
#                 cur_objt[o] = 1
#             object_mask_target.append([[cur_objt], [cur_objt]])
#         obj_target_tt = torch.FloatTensor(object_mask_target).squeeze().cuda()
#         return tmpl_target_tt, obj_target_tt
#
#     def generate_graph_mask(self, graph_infos):
#         assert len(graph_infos) == self.batch_size
#         mask_all = []
#         # TODO use graph dropout for masking here
#         for graph_info in graph_infos:
#             mask = [0] * len(self.vocab_act.keys())
#             # Case 1 (default): KG as mask
#             if self.params['masking'] == 'kg':
#                 # Uses the knowledge graph as the mask.
#                 graph_state = graph_info.graph_state
#                 ents = set()
#                 for u, v in graph_state.edges:
#                     ents.add(u)
#                     ents.add(v)
#                 # Build mask: only use those related to entities
#                 for ent in ents:
#                     for ent_word in ent.split():
#                         if ent_word[:self.max_word_length] in self.vocab_act_rev:
#                             idx = self.vocab_act_rev[ent_word[:self.max_word_length]]
#                             mask[idx] = 1
#                 if self.params['mask_dropout'] != 0:
#                     drop = random.sample(range(0, len(self.vocab_act.keys()) - 1),
#                                          int(self.params['mask_dropout'] * len(self.vocab_act.keys())))
#                     for i in drop:
#                         mask[i] = 1
#             # Case 2: interactive objects ground truth as the mask.
#             elif self.params['masking'] == 'interactive':
#                 # Uses interactive objects grount truth as the mask.
#                 for o in graph_info.objs:
#                     o = o[:self.max_word_length]
#                     if o in self.vocab_act_rev.keys() and o != '':
#                         mask[self.vocab_act_rev[o]] = 1
#                     if self.params['mask_dropout'] != 0:
#                         drop = random.sample(range(0, len(self.vocab_act.keys()) - 1),
#                                              int(self.params['mask_dropout'] * len(self.vocab_act.keys())))
#                         for i in drop:
#                             mask[i] = 1
#             # Case 3: no mask.
#             elif self.params['masking'] == 'none':
#                 # No mask at all.
#                 mask = [1] * len(self.vocab_act.keys())
#             else:
#                 assert False, 'Unrecognized masking {}'.format(self.params['masking'])
#             mask_all.append(mask)
#         return torch.BoolTensor(mask_all).cuda().detach()
#
#     def discount_reward(self, transitions, last_values):
#         returns, advantages = [], []
#         R = last_values.data
#         for t in reversed(range(len(transitions))):
#             _, _, values, rewards, done_masks, _, _, _, _, _, _ = transitions[t]
#             R = rewards + self.params['gamma'] * R * done_masks
#             adv = R - values
#             returns.append(R)
#             advantages.append(adv)
#         return returns[::-1], advantages[::-1]
#
#     def goexplore_train(self, obs, infos, graph_infos, max_steps, INTRINSIC_MOTIVTATION):
#         start = time.time()
#         transitions = []
#         if obs == None:
#             obs, infos, graph_infos = self.vec_env.go_reset()
#         for step in range(1, max_steps + 1):
#             self.total_steps += 1
#             tb.logkv('Step', self.total_steps)
#             obs_reps = np.array([g.ob_rep for g in graph_infos])
#             graph_mask_tt = self.generate_graph_mask(graph_infos)
#             graph_state_reps = [g.graph_state_rep for g in graph_infos]
#             # scores = [info['score'] for info in infos]
#             if self.params['reward_type'] == 'game_only':
#                 scores = [info['score'] for info in infos]
#             elif self.params['reward_type'] == 'IM_only':
#                 scores = np.array(
#                     [int(len(INTRINSIC_MOTIVTATION[i]) * self.params['intrinsic_motivation_factor']) for i in
#                      range(self.params['batch_size'])])
#             elif self.params['reward_type'] == 'game_and_IM':
#                 scores = np.array([infos[i]['score'] + (len(INTRINSIC_MOTIVTATION[i]) * (
#                             (infos[i]['score'] + self.params['epsilon']) / self.max_game_score)) for i in
#                                    range(self.params['batch_size'])])
#
#             tmpl_pred_tt, obj_pred_tt, dec_obj_tt, dec_tmpl_tt, value, dec_steps = self.model(
#                 obs_reps, scores, graph_state_reps, graph_mask_tt)
#             tb.logkv_mean('Value', value.mean().item())
#
#             # Log some of the predictions and ground truth values
#             topk_tmpl_probs, topk_tmpl_idxs = F.softmax(tmpl_pred_tt[0]).topk(5)
#             topk_tmpls = [self.template_generator.templates[t] for t in topk_tmpl_idxs.tolist()]
#             tmpl_pred_str = ', '.join(
#                 ['{} {:.3f}'.format(tmpl, prob) for tmpl, prob in zip(topk_tmpls, topk_tmpl_probs.tolist())])
#
#             admissible = [g.admissible_actions for g in graph_infos]
#             objs = [g.objs for g in graph_infos]
#             tmpl_gt_tt, obj_mask_gt_tt = self.generate_targets(admissible, objs)
#
#             gt_tmpls = [self.template_generator.templates[i] for i in
#                         tmpl_gt_tt[0].nonzero().squeeze().cpu().numpy().flatten().tolist()]
#             gt_objs = [self.vocab_act[i] for i in
#                        obj_mask_gt_tt[0, 0].nonzero().squeeze().cpu().numpy().flatten().tolist()]
#             log('TmplPred: {} GT: {}'.format(tmpl_pred_str, ', '.join(gt_tmpls)))
#             topk_o1_probs, topk_o1_idxs = F.softmax(obj_pred_tt[0, 0]).topk(5)
#             topk_o1 = [self.vocab_act[o] for o in topk_o1_idxs.tolist()]
#             o1_pred_str = ', '.join(['{} {:.3f}'.format(o, prob) for o, prob in zip(topk_o1, topk_o1_probs.tolist())])
#             graph_mask_str = [self.vocab_act[i] for i in
#                               graph_mask_tt[0].nonzero().squeeze().cpu().numpy().flatten().tolist()]
#             log('ObjtPred: {} GT: {} Mask: {}'.format(o1_pred_str, ', '.join(gt_objs), ', '.join(graph_mask_str)))
#
#             chosen_actions = self.decode_actions(dec_tmpl_tt, dec_obj_tt)
#
#             # Chooses random valid-actions to execute
#
#             obs, rewards, dones, infos, graph_infos = self.vec_env.go_step(chosen_actions)
#
#             edges = [set(graph_info.graph_state.edges) for graph_info in graph_infos]
#             size_updates = [0] * self.params['batch_size']
#             for i, s in enumerate(INTRINSIC_MOTIVTATION):
#                 orig_size = len(s)
#                 s.update(edges[i])
#                 size_updates[i] = len(s) - orig_size
#             rewards = list(rewards)
#             for i in range(self.params['batch_size']):
#                 if self.params['reward_type'] == 'IM_only':
#                     rewards[i] = size_updates[i] * self.params['intrinsic_motivation_factor']
#                 elif self.params['reward_type'] == 'game_and_IM':
#                     rewards[i] += size_updates[i] * self.params['intrinsic_motivation_factor']
#             rewards = tuple(rewards)
#
#             tb.logkv_mean('TotalStepsPerEpisode', sum([i['steps'] for i in infos]) / float(len(graph_infos)))
#             tb.logkv_mean('Valid', infos[0]['valid'])
#             log('Act: {}, Rew {}, Score {}, Done {}, Value {:.3f}'.format(
#                 chosen_actions[0], rewards[0], infos[0]['score'], dones[0], value[0].item()))
#             log('Obs: {}'.format(clean(obs[0])))
#             if dones[0]:
#                 log('Step {} EpisodeScore {}\n'.format(step, infos[0]['score']))
#             for done, info in zip(dones, infos):
#                 if done:
#                     tb.logkv_mean('EpisodeScore', info['score'])
#             rew_tt = torch.FloatTensor(rewards).cuda().unsqueeze(1)
#
#             done_mask_tt = (~torch.tensor(dones)).float().cuda().unsqueeze(1)
#             self.model.reset_hidden(done_mask_tt)
#             transitions.append((tmpl_pred_tt, obj_pred_tt, value, rew_tt,
#                                 done_mask_tt, tmpl_gt_tt, dec_tmpl_tt,
#                                 dec_obj_tt, obj_mask_gt_tt, graph_mask_tt, dec_steps))
#
#             if len(transitions) >= self.params['bptt']:
#                 tb.logkv('StepsPerSecond', float(step) / (time.time() - start))
#                 self.model.clone_hidden()
#                 obs_reps = np.array([g.ob_rep for g in graph_infos])
#                 graph_mask_tt = self.generate_graph_mask(graph_infos)
#                 graph_state_reps = [g.graph_state_rep for g in graph_infos]
#                 # scores = [info['score'] for info in infos]
#                 if self.params['reward_type'] == 'game_only':
#                     scores = [info['score'] for info in infos]
#                 elif self.params['reward_type'] == 'IM_only':
#                     scores = np.array(
#                         [int(len(INTRINSIC_MOTIVTATION[i]) * self.params['intrinsic_motivation_factor']) for i in
#                          range(self.params['batch_size'])])
#                 elif self.params['reward_type'] == 'game_and_IM':
#                     scores = np.array([infos[i]['score'] + (len(INTRINSIC_MOTIVTATION[i]) * (
#                                 (infos[i]['score'] + self.params['epsilon']) / self.max_game_score)) for i in
#                                        range(self.params['batch_size'])])
#                 _, _, _, _, next_value, _ = self.model(obs_reps, scores, graph_state_reps, graph_mask_tt)
#                 returns, advantages = self.discount_reward(transitions, next_value)
#                 log('Returns: ', ', '.join(['{:.3f}'.format(a[0].item()) for a in returns]))
#                 log('Advants: ', ', '.join(['{:.3f}'.format(a[0].item()) for a in advantages]))
#                 tb.logkv_mean('Advantage', advantages[-1].median().item())
#                 loss = self.update(transitions, returns, advantages)
#                 del transitions[:]
#                 self.model.restore_hidden()
#
#             if step % self.params['checkpoint_interval'] == 0:
#                 parameters = {'model': self.model}
#                 torch.save(parameters, os.path.join(self.params['output_dir'], 'qbert.pt'))
#
#         # self.vec_env.close_extras()
#         return obs, rewards, dones, infos, graph_infos, scores, chosen_actions, INTRINSIC_MOTIVTATION
#
#     def train(self, max_steps):
#         print("=== === === start training!!! === === ===")
#         start = time.time()
#         if self.params['training_type'] == 'chained':
#             self.log_file("BEGINNING OF TRAINING: patience={}, max_n_steps_back={}\n".format(self.params['patience'],
#                                                                                              self.params[
#                                                                                                  'buffer_size']))
#         frozen_policies = []
#         transitions = []
#         self.back_step = -1
#
#         previous_best_seen_score = float("-inf")
#         previous_best_step = 0
#         previous_best_state = None
#         previous_best_snapshot = None
#         previous_best_ACTUAL_score = 0
#         self.cur_reload_step = 0
#         force_reload = [False] * self.params['batch_size']
#         last_edges = None
#
#         self.valid_track = np.zeros(self.params['batch_size'])
#         self.stagnant_steps = 0
#
#         INTRINSIC_MOTIVTATION = [set() for i in range(self.params['batch_size'])]
#
#         obs, infos, graph_infos, env_str = self.vec_env.reset()
#         snap_obs = obs[0]
#         snap_info = infos[0]
#         snap_graph_reps = None
#         # print (obs)
#         # print (infos)
#         # print (graph_infos)
#         for step in range(1, max_steps + 1):
#             # Step 1: build model inputs
#             wallclock = time.time()
#
#             if any(force_reload) and self.params['training_type'] == 'chained':
#                 num_reload = force_reload.count(True)
#                 t_obs = np.array(obs)
#                 t_obs[force_reload] = [snap_obs] * num_reload
#                 obs = tuple(t_obs)
#
#                 t_infos = np.array(infos)
#                 t_infos[force_reload] = [snap_info] * num_reload
#                 infos = tuple(t_infos)
#
#                 t_graphs = list(graph_infos)
#                 # namedtuple gets lost in np.array
#                 t_updates = self.vec_env.load_from(self.cur_reload_state, force_reload, snap_graph_reps, snap_obs)
#                 for i in range(self.params['batch_size']):
#                     if force_reload[i]:
#                         t_graphs[i] = t_updates[i]
#                 graph_infos = tuple(t_graphs)
#
#                 force_reload = [False] * self.params['batch_size']
#
#             tb.logkv('Step', step)
#             obs_reps = np.array([g.ob_rep for g in graph_infos])
#             graph_mask_tt = self.generate_graph_mask(graph_infos)
#             graph_state_reps = [g.graph_state_rep for g in graph_infos]
#
#             if self.params['reward_type'] == 'game_only':
#                 scores = [info['score'] for info in infos]
#             elif self.params['reward_type'] == 'IM_only':
#                 scores = np.array(
#                     [int(len(INTRINSIC_MOTIVTATION[i]) * self.params['intrinsic_motivation_factor']) for i in
#                      range(self.params['batch_size'])])
#             elif self.params['reward_type'] == 'game_and_IM':
#                 scores = np.array([infos[i]['score'] + (len(INTRINSIC_MOTIVTATION[i]) * (
#                             (infos[i]['score'] + self.params['epsilon']) / self.max_game_score)) for i in
#                                    range(self.params['batch_size'])])
#             # print('!!!score', scores)
#             # print('graph_infos ===>', graph_infos)
#             graph_rep_1 = [g.graph_state_rep_1 for g in graph_infos]
#             graph_rep_2 = [g.graph_state_rep_2 for g in graph_infos]
#             graph_rep_3 = [g.graph_state_rep_3 for g in graph_infos]
#             graph_rep_4 = [g.graph_state_rep_4 for g in graph_infos]
#             # print('graph_rep_1 ===> ', graph_rep_1)
#             # print('graph_rep_2 ===> ', graph_rep_2)
#             # print('graph_rep_3 ===> ', graph_rep_3)
#             # Step 2: predict probs, actual items
#             tmpl_pred_tt, obj_pred_tt, dec_obj_tt, dec_tmpl_tt, value, dec_steps, output_gat, query_important = self.model(
#                 obs_reps,
#                 scores,
#                 graph_state_reps,
#                 graph_rep_1,
#                 graph_rep_2,
#                 graph_rep_3,
#                 graph_rep_4,
#                 graph_mask_tt)
#
#             # tmpl_pred_tt, obj_pred_tt, dec_obj_tt, dec_tmpl_tt, value, dec_steps = self.model(
#             #     obs_reps, scores, graph_state_reps, graph_mask_tt)
#             tb.logkv_mean('Value', value.mean().item())
#
#             # Step 3: Log the predictions and ground truth values
#             # Log the predictions and ground truth values
#             topk_tmpl_probs, topk_tmpl_idxs = F.softmax(tmpl_pred_tt[0]).topk(5)
#             topk_tmpls = [self.template_generator.templates[t] for t in topk_tmpl_idxs.tolist()]
#             tmpl_pred_str = ', '.join(
#                 ['{} {:.3f}'.format(tmpl, prob) for tmpl, prob in zip(topk_tmpls, topk_tmpl_probs.tolist())])
#
#             # Step 4: Generate the ground truth and object mask
#             admissible = [g.admissible_actions for g in graph_infos]
#             objs = [g.objs for g in graph_infos]
#             tmpl_gt_tt, obj_mask_gt_tt = self.generate_targets(admissible, objs)
#
#             # Step 5 Log template/object predictions/ground_truth
#             gt_tmpls = [self.template_generator.templates[i] for i in
#                         tmpl_gt_tt[0].nonzero().squeeze().cpu().numpy().flatten().tolist()]
#             gt_objs = [self.vocab_act[i] for i in
#                        obj_mask_gt_tt[0, 0].nonzero().squeeze().cpu().numpy().flatten().tolist()]
#             log('TmplPred: {} GT: {}'.format(tmpl_pred_str, ', '.join(gt_tmpls)))
#             topk_o1_probs, topk_o1_idxs = F.softmax(obj_pred_tt[0, 0]).topk(5)
#             topk_o1 = [self.vocab_act[o] for o in topk_o1_idxs.tolist()]
#             o1_pred_str = ', '.join(['{} {:.3f}'.format(o, prob) for o, prob in zip(topk_o1, topk_o1_probs.tolist())])
#             # graph_mask_str = [self.vocab_act[i] for i in graph_mask_tt[0].nonzero().squeeze().cpu().numpy().flatten().tolist()]
#             log('ObjtPred: {} GT: {}'.format(o1_pred_str, ', '.join(gt_objs)))  # , ', '.join(graph_mask_str)))
#
#             chosen_actions = self.decode_actions(dec_tmpl_tt, dec_obj_tt)
#
#             # stepclock = time.time()
#             # Step 6: Next step
#             obs, rewards, dones, infos, graph_infos, env_str = self.vec_env.step(chosen_actions)
#
#             # print('stepclock', time.time() - stepclock)
#             self.valid_track += [info['valid'] for info in infos]
#             self.stagnant_steps += 1
#             force_reload = list(dones)
#
#             edges = [set(graph_info.graph_state.edges) for graph_info in graph_infos]
#             size_updates = [0] * self.params['batch_size']
#             for i, s in enumerate(INTRINSIC_MOTIVTATION):
#                 orig_size = len(s)
#                 s.update(edges[i])
#                 size_updates[i] = len(s) - orig_size
#             rewards = list(rewards)
#             for i in range(self.params['batch_size']):
#                 if self.params['reward_type'] == 'IM_only':
#                     rewards[i] = size_updates[i] * self.params['intrinsic_motivation_factor']
#                 elif self.params['reward_type'] == 'game_and_IM':
#                     rewards[i] += size_updates[i] * self.params['intrinsic_motivation_factor']
#             rewards = tuple(rewards)
#
#             if last_edges:
#                 stayed_same = [1 if (len(edges[i] - last_edges[i]) <= self.params['kg_diff_threshold']) else 0 for i in
#                                range(self.params['batch_size'])]
#                 # print ("stayed_same: {}".format(stayed_same))
#             valid_kg_update = last_edges and sum(stayed_same) / self.params['batch_size'] > self.params[
#                 'kg_diff_batch_percentage']
#             last_edges = edges
#
#             snapshot = self.vec_env.get_snapshot()
#             real_scores = np.array([infos[i]['score'] for i in range(len(rewards))])
#
#             if self.params['reward_type'] == 'game_only':
#                 scores = [info['score'] for info in infos]
#             elif self.params['reward_type'] == 'IM_only':
#                 scores = np.array(
#                     [int(len(INTRINSIC_MOTIVTATION[i]) * self.params['intrinsic_motivation_factor']) for i in
#                      range(self.params['batch_size'])])
#             elif self.params['reward_type'] == 'game_and_IM':
#                 scores = np.array([infos[i]['score'] + (len(INTRINSIC_MOTIVTATION[i]) * (
#                             (infos[i]['score'] + self.params['epsilon']) / self.max_game_score)) for i in
#                                    range(self.params['batch_size'])])
#             cur_max_score_idx = np.argmax(scores)
#             if scores[cur_max_score_idx] > previous_best_seen_score and self.params[
#                 'training_type'] == 'chained':  # or valid_kg_update:
#                 print("New Reward Founded OR KG updated")
#                 previous_best_step = step
#                 previous_best_state = env_str[cur_max_score_idx]
#                 previous_best_seen_score = scores[cur_max_score_idx]
#                 previous_best_snapshot = snapshot[cur_max_score_idx]
#                 self.back_step = -1
#                 self.valid_track = np.zeros(self.params['batch_size'])
#                 self.stagnant_steps = 0
#                 print("\tepoch: {}".format(previous_best_step))
#                 print("\tnew score: {}".format(previous_best_seen_score))
#                 print("\tthis info: {}".format(infos[cur_max_score_idx]))
#                 self.log_file("New High Score Founded: step:{}, new_score:{}, infos:{}\n".format(previous_best_step,
#                                                                                                  previous_best_seen_score,
#                                                                                                  infos[
#                                                                                                      cur_max_score_idx]))
#
#             previous_best_ACTUAL_score = max(np.max(real_scores), previous_best_ACTUAL_score)
#             print("step {}: scores: {}, best_real_score: {}".format(step, scores, previous_best_ACTUAL_score))
#
#             tb.logkv_mean('TotalStepsPerEpisode', sum([i['steps'] for i in infos]) / float(len(graph_infos)))
#             tb.logkv_mean('Valid', infos[0]['valid'])
#             log('Act: {}, Rew {}, Score {}, Done {}, Value {:.3f}'.format(
#                 chosen_actions[0], rewards[0], infos[0]['score'], dones[0], value[0].item()))
#             log('Obs: {}'.format(clean(obs[0])))
#             if dones[0]:
#                 log('Step {} EpisodeScore {}\n'.format(step, infos[0]['score']))
#             for done, info in zip(dones, infos):
#                 if done:
#                     tb.logkv_mean('EpisodeScore', info['score'])
#
#             # Step 8: append into transitions
#             rew_tt = torch.FloatTensor(rewards).cuda().unsqueeze(1)
#             done_mask_tt = (~torch.tensor(dones)).float().cuda().unsqueeze(1)
#             self.model.reset_hidden(done_mask_tt)
#             transitions.append((tmpl_pred_tt, obj_pred_tt, value, rew_tt,
#                                 done_mask_tt, tmpl_gt_tt, dec_tmpl_tt,
#                                 dec_obj_tt, obj_mask_gt_tt, graph_mask_tt, dec_steps))
#
#             # Step 9: update model per 8 steps
#             if len(transitions) >= self.params['bptt']:
#                 tb.logkv('StepsPerSecond', float(step) / (time.time() - start))
#                 self.model.clone_hidden()
#                 obs_reps = np.array([g.ob_rep for g in graph_infos])
#                 graph_mask_tt = self.generate_graph_mask(graph_infos)
#                 graph_state_reps = [g.graph_state_rep for g in graph_infos]
#                 graph_rep_1 = [g.graph_state_rep_1 for g in graph_infos]
#                 graph_rep_2 = [g.graph_state_rep_2 for g in graph_infos]
#                 graph_rep_3 = [g.graph_state_rep_3 for g in graph_infos]
#                 graph_rep_4 = [g.graph_state_rep_4 for g in graph_infos]
#
#                 if self.params['reward_type'] == 'game_only':
#                     scores = [info['score'] for info in infos]
#                 elif self.params['reward_type'] == 'IM_only':
#                     scores = np.array(
#                         [int(len(INTRINSIC_MOTIVTATION[i]) * self.params['intrinsic_motivation_factor']) for i in
#                          range(self.params['batch_size'])])
#                 elif self.params['reward_type'] == 'game_and_IM':
#                     scores = np.array([infos[i]['score'] + (len(INTRINSIC_MOTIVTATION[i]) * (
#                                 (infos[i]['score'] + self.params['epsilon']) / self.max_game_score)) for i in
#                                        range(self.params['batch_size'])])
#
#                 _, _, _, _, next_value, _, output_gat, query_important = self.model(obs_reps,
#                                                        scores,
#                                                        graph_state_reps,
#                                                        graph_rep_1,
#                                                        graph_rep_2,
#                                                        graph_rep_3,
#                                                        graph_rep_4,
#                                                        graph_mask_tt)
#
#                 returns, advantages = self.discount_reward(transitions, next_value)
#                 log('Returns: ', ', '.join(['{:.3f}'.format(a[0].item()) for a in returns]))
#                 log('Advants: ', ', '.join(['{:.3f}'.format(a[0].item()) for a in advantages]))
#                 tb.logkv_mean('Advantage', advantages[-1].median().item())
#                 loss = self.update(transitions, returns, advantages)
#                 del transitions[:]
#                 self.model.restore_hidden()
#
#             if step % self.params['checkpoint_interval'] == 0:
#                 parameters = {'model': self.model}
#                 torch.save(parameters, os.path.join(self.params['output_dir'], 'qbert.pt'))
#
#             bottleneck = self.params['training_type'] == 'chained' and \
#                          ((self.stagnant_steps >= self.params['patience'] and not self.params['patience_valid_only']) or
#                           (self.params['patience_valid_only'] and sum(self.valid_track >= self.params['patience']) >=
#                            self.params['batch_size'] * self.params['patience_batch_factor']))
#             if bottleneck:
#                 print("Bottleneck detected at step: {}".format(step))
#                 # new_backstep += 1
#                 # new_back_step = (step - previous_best_step - self.params['patience']) // self.params['patience']
#                 self.back_step += 1
#                 if self.back_step == 0:
#                     self.vec_env.import_snapshot(previous_best_snapshot)
#                     cur_time = time.strftime("%Y%m%d-%H%M%S")
#                     torch.save(self.model.state_dict(), os.path.join(self.chkpt_path, '{}.pt'.format(cur_time)))
#                     frozen_policies.append((cur_time, previous_best_state))
#                     # INTRINSIC_MOTIVTATION= [set() for i in range(self.params['batch_size'])]
#                     self.log_file("Current model saved at: model/checkpoints/{}.pt\n".format(cur_time))
#                 self.model = QBERT(self.params, self.template_generator.templates, self.max_word_length,
#                                    self.vocab_act, self.vocab_act_rev, len(self.sp), gat=self.params['gat']).cuda()
#
#                 if self.back_step >= self.params['buffer_size']:
#                     print("Buffer exhausted. Finishing training")
#                     self.vec_env.close_extras()
#                     return
#                 print(previous_best_snapshot[-1 - self.back_step])
#                 snap_obs, snap_info, snap_graph_reps, self.cur_reload_state = previous_best_snapshot[
#                     -1 - self.back_step]
#                 print("Loading snapshot, infos: {}".format(snap_info))
#                 self.log_file("Loading snapshot, infos: {}\n".format(snap_info))
#                 self.cur_reload_step = previous_best_step
#                 force_reload = [True] * self.params['batch_size']
#                 self.valid_track = np.zeros(self.params['batch_size'])
#                 self.stagnant_steps = 0
#
#                 # print out observations here
#                 print("Current observations: {}".format([info['look'] for info in infos]))
#                 print("Previous_best_step: {}, step_back: {}".format(previous_best_step, self.back_step))
#                 self.log_file("Bottleneck Detected: step:{}, previous_best_step:{}, cur_step_back:{}\n".format(i,
#                                                                                                                previous_best_step,
#                                                                                                                self.back_step))
#                 self.log_file("Current observations: {}\n".format([info['look'] for info in infos]))
#             # exit()
#
#         self.vec_env.close_extras()
#
#     def update(self, transitions, returns, advantages):
#         assert len(transitions) == len(returns) == len(advantages)
#         loss = 0
#         for trans, ret, adv in zip(transitions, returns, advantages):
#             tmpl_pred_tt, obj_pred_tt, value, _, _, tmpl_gt_tt, dec_tmpl_tt, \
#             dec_obj_tt, obj_mask_gt_tt, graph_mask_tt, dec_steps = trans
#
#             # Supervised Template Loss
#             tmpl_probs = F.softmax(tmpl_pred_tt, dim=1)
#             template_loss = self.params['template_coeff'] * self.loss_fn1(tmpl_probs, tmpl_gt_tt)
#
#             # Supervised Object Loss
#             if self.params['batch_size'] == 1:
#                 object_mask_target = obj_mask_gt_tt.unsqueeze(0).permute((1, 0, 2))
#             else:
#                 object_mask_target = obj_mask_gt_tt.permute((1, 0, 2))
#             obj_probs = F.softmax(obj_pred_tt, dim=2)
#             object_mask_loss = self.params['object_coeff'] * self.loss_fn1(obj_probs, object_mask_target)
#
#             # Build the object mask
#             o1_mask, o2_mask = [0] * self.batch_size, [0] * self.batch_size
#             for d, st in enumerate(dec_steps):
#                 if st > 1:
#                     o1_mask[d] = 1
#                     o2_mask[d] = 1
#                 elif st == 1:
#                     o1_mask[d] = 1
#             o1_mask = torch.FloatTensor(o1_mask).cuda()
#             o2_mask = torch.FloatTensor(o2_mask).cuda()
#
#             # Policy Gradient Loss
#             policy_obj_loss = torch.FloatTensor([0]).cuda()
#             cnt = 0
#             for i in range(self.batch_size):
#                 if dec_steps[i] >= 1:
#                     cnt += 1
#                     batch_pred = obj_pred_tt[0, i, graph_mask_tt[i]]
#                     action_log_probs_obj = F.log_softmax(batch_pred, dim=0)
#                     dec_obj_idx = dec_obj_tt[0, i].item()
#                     graph_mask_list = graph_mask_tt[i].nonzero().squeeze().cpu().numpy().flatten().tolist()
#                     idx = graph_mask_list.index(dec_obj_idx)
#                     log_prob_obj = action_log_probs_obj[idx]
#                     policy_obj_loss += -log_prob_obj * adv[i].detach()
#             if cnt > 0:
#                 policy_obj_loss /= cnt
#             tb.logkv_mean('PolicyObjLoss', policy_obj_loss.item())
#             log_probs_obj = F.log_softmax(obj_pred_tt, dim=2)
#
#             log_probs_tmpl = F.log_softmax(tmpl_pred_tt, dim=1)
#             action_log_probs_tmpl = log_probs_tmpl.gather(1, dec_tmpl_tt).squeeze()
#
#             policy_tmpl_loss = (-action_log_probs_tmpl * adv.detach().squeeze()).mean()
#             tb.logkv_mean('PolicyTemplateLoss', policy_tmpl_loss.item())
#
#             policy_loss = policy_tmpl_loss + policy_obj_loss
#
#             value_loss = self.params['value_coeff'] * self.loss_fn3(value, ret)
#             tmpl_entropy = -(tmpl_probs * log_probs_tmpl).mean()
#             tb.logkv_mean('TemplateEntropy', tmpl_entropy.item())
#             object_entropy = -(obj_probs * log_probs_obj).mean()
#             tb.logkv_mean('ObjectEntropy', object_entropy.item())
#             # Minimizing entropy loss will lead to increased entropy
#             entropy_loss = self.params['entropy_coeff'] * -(tmpl_entropy + object_entropy)
#
#             loss += template_loss + object_mask_loss + value_loss + entropy_loss + policy_loss
#
#         tb.logkv('Loss', loss.item())
#         tb.logkv('TemplateLoss', template_loss.item())
#         tb.logkv('ObjectLoss', object_mask_loss.item())
#         tb.logkv('PolicyLoss', policy_loss.item())
#         tb.logkv('ValueLoss', value_loss.item())
#         tb.logkv('EntropyLoss', entropy_loss.item())
#         tb.dumpkvs()
#         loss.backward()
#
#         # Compute the gradient norm
#         grad_norm = 0
#         for p in list(filter(lambda p: p.grad is not None, self.model.parameters())):
#             grad_norm += p.grad.data.norm(2).item()
#         tb.logkv('UnclippedGradNorm', grad_norm)
#
#         nn.utils.clip_grad_norm_(self.model.parameters(), self.params['clip'])
#
#         # Clipped Grad norm
#         grad_norm = 0
#         for p in list(filter(lambda p: p.grad is not None, self.model.parameters())):
#             grad_norm += p.grad.data.norm(2).item()
#         tb.logkv('ClippedGradNorm', grad_norm)
#
#         self.optimizer.step()
#         self.optimizer.zero_grad()
#         return loss
#
#     def decode_actions(self, decoded_templates, decoded_objects):
#         '''
#         Returns string representations of the given template actions.
#
#         :param decoded_template: Tensor of template indices.
#         :type decoded_template: Torch tensor of size (Batch_size x 1).
#         :param decoded_objects: Tensor of o1, o2 object indices.
#         :type decoded_objects: Torch tensor of size (2 x Batch_size x 1).
#
#         '''
#         decoded_actions = []
#         for i in range(self.batch_size):
#             decoded_template = decoded_templates[i].item()
#             decoded_object1 = decoded_objects[0][i].item()
#             decoded_object2 = decoded_objects[1][i].item()
#             decoded_action = self.tmpl_to_str(decoded_template, decoded_object1, decoded_object2)
#             decoded_actions.append(decoded_action)
#         return decoded_actions
#
#     def tmpl_to_str(self, template_idx, o1_id, o2_id):
#         """ Returns a string representation of a template action. """
#         template_str = self.template_generator.templates[template_idx]
#         holes = template_str.count('OBJ')
#         assert holes <= 2
#         if holes <= 0:
#             return template_str
#         elif holes == 1:
#             return template_str.replace('OBJ', self.vocab_act[o1_id])
#         else:
#             return template_str.replace('OBJ', self.vocab_act[o1_id], 1) \
#                 .replace('OBJ', self.vocab_act[o2_id], 1)
