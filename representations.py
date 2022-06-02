# Different graphs
#  graph_state: full new kg
#  graph_state_1: room connectivity (history included)
#  graph_state_2: what's in current room
#  graph_state_3: your inventory
#  graph_state_4: remove you related nodes (history included)
#  graph_state_5_mask (not used): intersection of kg1, kg2 and kg3

import os
import networkx as nx
import numpy as np
from fuzzywuzzy import fuzz
from jericho.util import clean
from jericho import defines
import extraction_api
import copy
import torch
from nltk_gen import nltk_gen

# from extraction import kgextraction


class StateAction(object):
    def __init__(
        self,
        spm,
        vocab,
        vocab_rev,
        tsv_file,
        max_word_len,
        askbert,
        attr_file,
        subKG_type="SHA",
    ):
        self.nltk = nltk_gen()
        self.graph_state = nx.DiGraph()
        self.max_word_len = max_word_len
        self.graph_state_rep = []
        self.visible_state = ""
        self.drqa_input = ""
        self.vis_pruned_actions = []
        self.pruned_actions_rep = []
        self.sp = spm
        self.vocab_act = vocab
        self.vocab_act_rev = vocab_rev
        vocabs = self.load_vocab_kge(tsv_file)
        self.vocab_kge, self.vocab_kgr = vocabs["entity"], vocabs["relation"]
        self.context_attr = self.load_attributes(attr_file)
        # print(self.context_attr)
        self.adj_matrix = np.array(
            [np.zeros((len(self.vocab_kge), len(self.vocab_kge)))] * len(self.vocab_kgr)
        )
        self.room = ""

        # =====Becky=====-#
        self.graph_state_1 = nx.DiGraph()  # Need to track room connectivity
        self.graph_state_2 = None
        self.graph_state_3 = None
        self.graph_state_4 = None
        self.graph_state_5_mask = None
        self.graph_state_rep_1 = []
        self.graph_state_rep_2 = []
        self.graph_state_rep_3 = []
        self.graph_state_rep_4 = []
        self.graph_state_rep_5_mask = []
        self.adj_matrix_1 = np.zeros((len(self.vocab_kge), len(self.vocab_kge)))
        self.adj_matrix_2 = np.zeros((len(self.vocab_kge), len(self.vocab_kge)))
        self.adj_matrix_3 = np.zeros((len(self.vocab_kge), len(self.vocab_kge)))
        self.adj_matrix_4 = np.zeros((len(self.vocab_kge), len(self.vocab_kge)))
        self.adj_matrix_5_mask = np.zeros((len(self.vocab_kge), len(self.vocab_kge)))
        self.subKG_type = subKG_type
        # =====END=====#

        self.askbert = askbert
        self.ABBRV_DICT = {
            "n": "north",
            "s": "south",
            "w": "west",
            "e": "east",
            "d": "down",
            "u": "up",
            "g": "again",
            "ne": "northeast",
            "nw": "northwest",
            "se": "southeast",
            "sw": "southwest",
        }
        self.MOVE_ACTIONS = "north/south/west/east/northwest/southwest/northeast/southeast/up/down/enter/exit".split(
            "/"
        )
        askbert_args = {
            "input_text": "",
            "length": 10,
            "batch_size": 1,
            "temperature": 1,
            "model_name": "117M",
            "seed": 0,
            "nsamples": 10,
            "cutoffs": "6 7 5",
            "write_sfdp": False,
            "random": False,
        }
        # self.extraction = kgextraction.World(askbert_args)
        self.ct = 0

    def visualize(self):
        import matplotlib.pyplot as plt

        pos = nx.kamada_kawai_layout(self.graph_state)
        edge_labels = {
            e: self.graph_state.edges[e]["rel"] for e in self.graph_state.edges
        }
        triples = []
        for k, v in edge_labels.items():
            triples.append((k[0], v, k[1]))
        # print(triples)
        # print(len(edge_labels.keys()), edge_labels)
        nx.draw_networkx_edge_labels(self.graph_state, pos, edge_labels)
        nx.draw(
            self.graph_state, pos=pos, with_labels=True, node_size=2000, font_size=10
        )
        plt.savefig(str(self.ct) + ".pdf")
        self.ct += 1
        # plt.show()

    def load_vocab_kge(self, tsv_file):
        ent = {}
        # alle = []
        with open(tsv_file, "r") as f:
            for line in f:
                e, eid = line.split("\t")
                ent[str(e.strip())] = int(eid.strip())
                # alle.append(str(e.strip()))
        # print(len(ent), len(alle), ent.keys(), alle)
        rel_path = os.path.dirname(tsv_file)
        rel_name = os.path.join(rel_path, "relation2id.tsv")
        rel = {}
        with open(rel_name, "r") as f:
            for line in f:
                r, rid = line.split("\t")
                rel[r.strip()] = int(rid.strip())
        return {"entity": ent, "relation": rel}

    def load_attributes(self, attr_file):
        context_attr = ""
        attr_file = "./attrs/" + attr_file + "_attr.txt"
        if os.path.isfile(attr_file):
            with open(attr_file, "r") as f:
                context_attr = str(f.read())
        context_attr = (
            "talkable, seen, lieable, enterable, nodwarf, indoors, visited, handed, lockable, surface, thing, "
            "water_room, unlock, lost, afflicted, is_treasure, converse, mentioned, male, npcworn, no_article, "
            "relevant, scored, queryable, town, pluggable, happy, is_followable, legible, multitude, burning, "
            "room, clothing, underneath, ward_area, little, intact, animate, bled_in, supporter, readable, "
            "openable, near, nonlocal, door, plugged, sittable, toolbit, vehicle, light, lens_searchable, "
            "open, familiar, is_scroll, aimable, takeable, static, unique, concealed, vowelstart, alcoholic, "
            "bodypart, general, is_spell, full, dry_land, pushable, known, proper, inside, clean, "
            "ambiguously_plural, container, edible, treasure, can_plug, weapon, is_arrow, insubstantial, "
            "pluralname, transparent, is_coin, air_room, scenery, on, is_spell_book, burnt, burnable, "
            "auto_searched, locked, switchable, absent, rockable, beenunlocked, progressing, severed, worn, "
            "windy, stone, random, neuter, legible, female, asleep, wiped"
        )

        return context_attr

    def update_state(
        self, visible_state, inventory_state, objs, prev_act=None, cache=None
    ):
        # =====Becky=====#
        if self.subKG_type == "SHA":
            # Step 1: Build a copy of past KG (full)
            graph_copy = self.graph_state.copy()
            prev_room = self.room
            prev_room_subgraph = None
            con_cs = [
                graph_copy.subgraph(c)
                for c in nx.weakly_connected_components(graph_copy)
            ]
            for con_c in con_cs:
                for node in con_c.nodes:
                    node = set(str(node).split())
                    if set(prev_room.split()).issubset(node):
                        prev_room_subgraph = nx.induced_subgraph(
                            graph_copy, con_c.nodes
                        )
            # Step 2: Bemove old ones with "you" --> past KG without "you"
            for edge in self.graph_state.edges:
                if "you" in edge[0]:
                    graph_copy.remove_edge(*edge)
            self.graph_state = graph_copy
            # Keep room connectivity only, remove "you"
            # <you, in, room>, <room, connect, room> --> <room, connect, room>
            graph_copy_1 = self.graph_state_1.copy()
            for edge in self.graph_state_1.edges:
                if "you" in edge[0]:
                    graph_copy_1.remove_edge(*edge)
            self.graph_state_1 = graph_copy_1
            # Step 3: Reinitialize sub-KG
            self.graph_state_2 = nx.DiGraph()  # re-init
            self.graph_state_3 = nx.DiGraph()  # re-init
            self.graph_state_4 = graph_copy.copy()  # Just past information
            # Preprocess visible state --> get sents
            visible_state = visible_state.split("\n")
            room = visible_state[0]
            visible_state = clean(" ".join(visible_state[1:]))
            self.visible_state = str(visible_state)
            # =====END=====#

            self.visible_state = visible_state
            prev_room = self.room
            add_triples = set()
            remove_triples = set()
            add_triples.add(("you", "is", "you"))

            if cache is not None:
                entities = cache
            else:
                entities = extraction_api.call_askbert(
                    self.visible_state + " [atr] " + self.context_attr,
                    self.askbert,
                    self.context_attr != "",
                )
                if entities is None:
                    self.askbert /= 1.5
                    return [], None
                entities = entities["entities"]
            # print('entities', entities)

            # =====Becky=====#
            dirs = [
                "north",
                "south",
                "east",
                "west",
                "southeast",
                "southwest",
                "northeast",
                "northwest",
                "up",
                "down",
            ]
            in_aliases = [
                "are in",
                "are facing",
                "are standing",
                "are behind",
                "are above",
                "are below",
                "are in front",
            ]
            # Update graph, "rules" are new triples to be added
            # Add two rule lists for "you" and "woyou"
            rules_1 = []  # <you,in>, <room,connect>
            rules_2 = []  # <you,in>, <room,have>
            rules_3 = []  # <you,have>
            rules = []
            in_rl = []
            in_flag = False
            # =======END======#

            # Location mappings
            if len(entities["location"]) != 0:
                self.room = entities["location"][0]

            if len(entities["location"]) == 0:
                self.room = ""

            if self.room != "":
                add_triples.add(("you", "in", self.room))
                remove_triples.add(("you", "in", prev_room))
                # =====Becky=====#
                in_rl.append(("you", "in", self.room))  # <you, in, >
                cur_t = in_rl[0]
                for h, r, t in in_rl:
                    if set(cur_t[2].split()).issubset(set(t.split())):
                        cur_t = h, r, t
                rules.append(cur_t)
                rules_1.append(cur_t)
                rules_2.append(cur_t)
                # ======END======#

                if prev_act.lower() in self.MOVE_ACTIONS:
                    add_triples.add((self.room, prev_act, prev_room))

                if prev_act.lower() in self.ABBRV_DICT.keys():
                    prev_act = defines.ABBRV_DICT[prev_act.lower()]
                    add_triples.add((self.room, prev_act, prev_room))

                surr_objs = entities["object_surr"]
                for s in surr_objs:
                    add_triples.add((s, "in", self.room))
                    # =====Becky=====#
                    rules.append((s, "in", self.room))
                    rules_2.append((s, "in", self.room))
                    # ======END======#
                    if self.graph_state.has_edge("you", s):
                        remove_triples.add(("you", "have", s))

                inv_objs = entities["objs_inv"]
                for i in inv_objs:
                    add_triples.add(("you", "have", i))
                    # =====Becky=====#
                    rules.append(("you", "have", i))
                    rules_3.append(("you", "have", i))
                    # ======END======#
                    if self.graph_state.has_edge(i, self.room):
                        remove_triples.add((i, "in", self.room))

                attributes = entities["attributes"]
                for o in inv_objs + surr_objs:
                    if o in attributes.keys():
                        a_curr = attributes[o]
                        for a in a_curr:
                            add_triples.add((o, "is", a))

            for rule in add_triples:
                u = "_".join(str(rule[0]).split()).lower()
                v = "_".join(str(rule[2]).split()).lower()
                r = "_".join(str(rule[1]).split()).lower()
                if (
                    u in self.vocab_kge.keys()
                    and v in self.vocab_kge.keys()
                    and r in self.vocab_kgr.keys()
                ):
                    self.graph_state.add_edge(u, v, rel=r)
            for rule in remove_triples:
                u = "_".join(str(rule[0]).split()).lower()
                v = "_".join(str(rule[2]).split()).lower()
                if u in self.vocab_kge.keys() and v in self.vocab_kge.keys():
                    if self.graph_state.has_edge(u, v):
                        # print("REMOVE", (u, v))
                        self.graph_state.remove_edge(u, v)

            # =====Becky=====#
            # build graph_state_1
            for rule in rules_1:
                u = "_".join(str(rule[0]).split())
                v = "_".join(str(rule[2]).split())
                if u in self.vocab_kge.keys() and v in self.vocab_kge.keys():
                    if u != "it" and v != "it":
                        self.graph_state_1.add_edge(rule[0], rule[2], rel=rule[1])
            # build graph_state_5_mask
            self.graph_state_5_mask = self.graph_state_1.copy()
            # build graph_state_2 (and graph_state_5_mask)
            for rule in rules_2:
                u = "_".join(str(rule[0]).split())
                v = "_".join(str(rule[2]).split())
                if u in self.vocab_kge.keys() and v in self.vocab_kge.keys():
                    if u != "it" and v != "it":
                        self.graph_state_2.add_edge(rule[0], rule[2], rel=rule[1])
                        self.graph_state_5_mask.add_edge(rule[0], rule[2], rel=rule[1])
            # build graph_state_3 (and graph_state_5_mask)
            for rule in rules_3:
                u = "_".join(str(rule[0]).split())
                v = "_".join(str(rule[2]).split())
                if u in self.vocab_kge.keys() and v in self.vocab_kge.keys():
                    if u != "it" and v != "it":
                        self.graph_state_3.add_edge(rule[0], rule[2], rel=rule[1])
                        self.graph_state_5_mask.add_edge(rule[0], rule[2], rel=rule[1])
                        # ======END======#
            # self.visualize()

        elif self.subKG_type == "Full":
            self.visible_state = visible_state
            prev_room = self.room
            add_triples = set()
            remove_triples = set()
            add_triples.add(("you", "is", "you"))

            if cache is not None:
                entities = cache
            else:
                entities = extraction_api.call_askbert(
                    self.visible_state + " [atr] " + self.context_attr,
                    self.askbert,
                    self.context_attr != "",
                )
                if entities is None:
                    self.askbert /= 1.5
                    return [], None
                entities = entities["entities"]

            # Location mappings
            if len(entities["location"]) != 0:
                self.room = entities["location"][0]

            if len(entities["location"]) == 0:
                self.room = ""

            if self.room != "":
                add_triples.add(("you", "in", self.room))
                remove_triples.add(("you", "in", prev_room))

                if prev_act.lower() in self.MOVE_ACTIONS:
                    add_triples.add((self.room, prev_act, prev_room))

                if prev_act.lower() in self.ABBRV_DICT.keys():
                    prev_act = defines.ABBRV_DICT[prev_act.lower()]
                    add_triples.add((self.room, prev_act, prev_room))

                surr_objs = entities["object_surr"]
                for s in surr_objs:
                    add_triples.add((s, "in", self.room))
                    if self.graph_state.has_edge("you", s):
                        remove_triples.add(("you", "have", s))

                inv_objs = entities["objs_inv"]
                for i in inv_objs:
                    add_triples.add(("you", "have", i))
                    if self.graph_state.has_edge(i, self.room):
                        remove_triples.add((i, "in", self.room))

                attributes = entities["attributes"]
                for o in inv_objs + surr_objs:
                    if o in attributes.keys():
                        a_curr = attributes[o]
                        for a in a_curr:
                            add_triples.add((o, "is", a))

            for rule in add_triples:
                u = "_".join(str(rule[0]).split()).lower()
                v = "_".join(str(rule[2]).split()).lower()
                r = "_".join(str(rule[1]).split()).lower()
                if (
                    u in self.vocab_kge.keys()
                    and v in self.vocab_kge.keys()
                    and r in self.vocab_kgr.keys()
                ):
                    self.graph_state.add_edge(u, v, rel=r)
            for rule in remove_triples:
                u = "_".join(str(rule[0]).split()).lower()
                v = "_".join(str(rule[2]).split()).lower()
                if u in self.vocab_kge.keys() and v in self.vocab_kge.keys():
                    if self.graph_state.has_edge(u, v):
                        # print("REMOVE", (u, v))
                        self.graph_state.remove_edge(u, v)

            # =====Becky=====#
            self.graph_state_1 = self.graph_state.copy()  # re-init
            self.graph_state_2 = self.graph_state.copy()  # re-init
            self.graph_state_3 = self.graph_state.copy()  # re-init
            self.graph_state_4 = self.graph_state.copy()  # re-init
            # self.visualize()

        else:  # our new 4 subgraph.
            # print('INININ')
            # Step 1: Reinitialize sub-KG
            self.graph_state_1 = nx.DiGraph()  # re-init
            self.graph_state_2 = nx.DiGraph()  # re-init
            self.graph_state_3 = nx.DiGraph()  # re-init
            self.graph_state_4 = nx.DiGraph()  # re-init
            self.graph_state_5_mask = nx.DiGraph()  # re-init
            self.visible_state = visible_state
            prev_room = self.room
            add_triples = set()
            remove_triples = set()
            add_triples.add(("you", "is", "you"))

            if cache is not None:
                entities = cache
            else:
                entities = extraction_api.call_askbert(
                    self.visible_state + " [atr] " + self.context_attr,
                    self.askbert,
                    self.context_attr != "",
                )
                if entities is None:
                    self.askbert /= 1.5
                    return [], None
                entities = entities["entities"]
                # print('visible_state=>', visible_state)
            # print('entities=>', entities)

            # =====Becky=====#
            dirs = [
                "north",
                "south",
                "east",
                "west",
                "southeast",
                "southwest",
                "northeast",
                "northwest",
                "up",
                "down",
            ]
            in_aliases = [
                "are in",
                "are facing",
                "are standing",
                "are behind",
                "are above",
                "are below",
                "are in front",
            ]
            # Update graph, "rules" are new triples to be added
            # Add two rule lists for "you" and "woyou"
            rules_1 = []  # <you,in>, <room,connect>
            rules_2 = []  # <you,in>, <room,have>
            rules_3 = []  # <you,have>
            rules_4 = []
            rules = []
            in_rl = []
            in_flag = False
            # =======END======#

            # Location mappings
            if len(entities["location"]) != 0:
                self.room = entities["location"][0]

            if len(entities["location"]) == 0:
                self.room = ""

            if self.room != "":
                add_triples.add(("you", "in", self.room))
                remove_triples.add(("you", "in", prev_room))
                # =====Becky=====#
                in_rl.append(("you", "in", self.room))  # <you, in, >
                rules.append(("you", "in", self.room))
                rules_3.append(("you", "in", self.room))  # __ 'in' __
                # ======END======#

                if prev_act.lower() in self.MOVE_ACTIONS:
                    add_triples.add((self.room, prev_act, prev_room))
                    rules.append((self.room, prev_act, prev_room))
                    rules_4.append((self.room, prev_act, prev_room))  # others

                if prev_act.lower() in self.ABBRV_DICT.keys():
                    prev_act = defines.ABBRV_DICT[prev_act.lower()]
                    add_triples.add((self.room, prev_act, prev_room))
                    rules.append((self.room, prev_act, prev_room))
                    rules_4.append((self.room, prev_act, prev_room))  # others

                surr_objs = entities["object_surr"]
                for s in surr_objs:
                    add_triples.add((s, "in", self.room))
                    # =====Becky=====#
                    rules.append((s, "in", self.room))
                    rules_3.append((s, "in", self.room))
                    # ======END======#
                    if self.graph_state.has_edge("you", s):
                        remove_triples.add(("you", "have", s))

                inv_objs = entities["objs_inv"]
                for i in inv_objs:
                    add_triples.add(("you", "have", i))
                    # =====Becky=====#
                    rules.append(("you", "have", i))
                    rules_2.append(("you", "have", i))  # 'you' 'have' __
                    # ======END======#
                    if self.graph_state.has_edge(i, self.room):
                        remove_triples.add((i, "in", self.room))

                attributes = entities["attributes"]
                for o in inv_objs + surr_objs:
                    if o in attributes.keys():
                        a_curr = attributes[o]
                        for a in a_curr:
                            add_triples.add((o, "is", a))
                            rules_1.append((o, "is", a))  # __ 'is' __ (Attr of objects)

            for rule in add_triples:
                u = "_".join(str(rule[0]).split()).lower()
                v = "_".join(str(rule[2]).split()).lower()
                r = "_".join(str(rule[1]).split()).lower()
                if (
                    u in self.vocab_kge.keys()
                    and v in self.vocab_kge.keys()
                    and r in self.vocab_kgr.keys()
                ):
                    self.graph_state.add_edge(u, v, rel=r)
            for rule in remove_triples:
                u = "_".join(str(rule[0]).split()).lower()
                v = "_".join(str(rule[2]).split()).lower()
                if u in self.vocab_kge.keys() and v in self.vocab_kge.keys():
                    if self.graph_state.has_edge(u, v):
                        self.graph_state.remove_edge(u, v)

            # =====Becky=====#
            # build sub graph_states

            for idx, (graph_state, rules) in enumerate(
                [
                    (self.graph_state_1, rules_1),
                    (self.graph_state_2, rules_2),
                    (self.graph_state_3, rules_3),
                    (self.graph_state_4, rules_4),
                ]
            ):
                # print(idx+1, ' => ', rules)
                for rule in rules:
                    graph_state.add_edge(rule[0], rule[2], rel=rule[1])
            self.graph_state_5_mask = self.graph_state_5_mask.copy()
        # print('self.graph_state_1 =>', self.graph_state_1.edges)
        # print('self.graph_state_2 = >', self.graph_state_2.edges)
        # print('self.graph_state_3 =>', self.graph_state_3.edges)
        # print('self.graph_state_4 = >', self.graph_state_4.edges)
        # print('=' * 5)
        return add_triples, entities

    def get_state_rep_kge(self):
        ret = []
        self.adj_matrix = np.zeros((len(self.vocab_kge), len(self.vocab_kge)))
        for u, v in self.graph_state.edges:
            u = "_".join(str(u).split())
            v = "_".join(str(v).split())
            # print('u, v', u, v)
            if u not in self.vocab_kge.keys() or v not in self.vocab_kge.keys():
                if len(u.split()) == 1 and len(v.split()) == 1:
                    pass
                else:
                    if len(u.split()) > 1 and len(v.split()) == 1:
                        res, _ = self.nltk.noun_adj_adv_find(u, u)
                        for _u in res:
                            u_idx = self.vocab_kge[_u]
                            v_idx = self.vocab_kge[v]
                            # print('u_idx,', u_idx, v_idx)
                            self.adj_matrix[u_idx][v_idx] = 1
                            ret.append(self.vocab_kge[_u])
                            ret.append(self.vocab_kge[v])
                    elif len(v.split()) > 1 and len(u.split()) == 1:
                        res, _ = self.nltk.noun_adj_adv_find(v, v)
                        for _v in res:
                            u_idx = self.vocab_kge[u]
                            v_idx = self.vocab_kge[_v]
                            # print('u_idx,', u_idx, v_idx)
                            self.adj_matrix[u_idx][v_idx] = 1
                            ret.append(self.vocab_kge[u])
                            ret.append(self.vocab_kge[_v])
                    else:
                        res_u, _ = self.nltk.noun_adj_adv_find(u, u)
                        res_v, _ = self.nltk.noun_adj_adv_find(v, v)
                        for _u in res:
                            for _v in res:
                                u_idx = self.vocab_kge[_u]
                                v_idx = self.vocab_kge[_v]
                                # print('u_idx,', u_idx, v_idx)
                                self.adj_matrix[u_idx][v_idx] = 1
                                ret.append(self.vocab_kge[_u])
                                ret.append(self.vocab_kge[_v])
                break
            u_idx = self.vocab_kge[u]
            v_idx = self.vocab_kge[v]
            # print('u_idx,', u_idx, v_idx)
            self.adj_matrix[u_idx][v_idx] = 1
            ret.append(self.vocab_kge[u])
            ret.append(self.vocab_kge[v])
        return list(set(ret))

    # def get_state_rep_kge(self):
    #     ret = []
    #     self.adj_matrix = np.zeros((len(self.vocab_kge), len(self.vocab_kge)))
    #     # print('graph', self.graph_state.edges)
    #     for u, v in self.graph_state.edges:
    #         u = '_'.join(str(u).split()).lower()
    #         v = '_'.join(str(v).split()).lower()
    #         # print('nodes are => ', u, v)
    #         if u in self.vocab_kge.keys() and v in self.vocab_kge.keys():
    #             u_idx = self.vocab_kge[u]
    #             v_idx = self.vocab_kge[v]
    #             # print('idx are => ', u_idx, v_idx)
    #             self.adj_matrix[u_idx][v_idx] = 1
    #             ret.append(self.vocab_kge[u])
    #             ret.append(self.vocab_kge[v])
    #         elif u in self.vocab_kge.keys() and v not in self.vocab_kge.keys():
    #             v = v.split('_')
    #             u_idx = self.vocab_kge[u]
    #             for _v in v:
    #                 if _v in self.vocab_kge.keys():
    #                     v_idx = self.vocab_kge[_v]
    #                     # print('idx are => ', u_idx, v_idx)
    #                     self.adj_matrix[u_idx][v_idx] = 1
    #                     ret.append(self.vocab_kge[u])
    #                     ret.append(self.vocab_kge[v])
    #         elif v in self.vocab_kge.keys() and u not in self.vocab_kge.keys():
    #             u = u.split('_')
    #             v_idx = self.vocab_kge[v]
    #             for _u in u:
    #                 if _u in self.vocab_kge.keys():
    #                     u_idx = self.vocab_kge[_u]
    #                     # print('idx are => ', u_idx, v_idx)
    #                     self.adj_matrix[u_idx][v_idx] = 1
    #                     ret.append(self.vocab_kge[u])
    #                     ret.append(self.vocab_kge[v])
    #         else:
    #             pass
    #
    #     return list(set(ret))

    def get_state_rep_kge_1(self):
        ret = []
        self.adj_matrix_1 = np.zeros((len(self.vocab_kge), len(self.vocab_kge)))
        for u, v in self.graph_state_1.edges:
            # print('>>>', u, v)
            u = "_".join(str(u).split()).lower()
            v = "_".join(str(v).split()).lower()
            # print('nodes are => ', u, v)
            if u in self.vocab_kge.keys() and v in self.vocab_kge.keys():
                u_idx = self.vocab_kge[u]
                v_idx = self.vocab_kge[v]
                # print('idx are => ', u_idx, v_idx)
                self.adj_matrix_1[u_idx][v_idx] = 1
                ret.append(self.vocab_kge[u])
                ret.append(self.vocab_kge[v])

            # elif u in self.vocab_kge.keys() and v not in self.vocab_kge.keys():
            #     v = v.split('_')
            #     u_idx = self.vocab_kge[u]
            #     for _v in v:
            #         if _v in self.vocab_kge.keys():
            #             v_idx = self.vocab_kge[_v]
            #             # print('idx are => ', u_idx, v_idx)
            #             self.adj_matrix_1[u_idx][v_idx] = 1
            #             ret.append(self.vocab_kge[u])
            #             ret.append(self.vocab_kge[v])
            # elif v in self.vocab_kge.keys() and u not in self.vocab_kge.keys():
            #     u = u.split('_')
            #     v_idx = self.vocab_kge[v]
            #     for _u in u:
            #         if _u in self.vocab_kge.keys():
            #             u_idx = self.vocab_kge[_u]
            #             # print('idx are => ', u_idx, v_idx)
            #             self.adj_matrix_1[u_idx][v_idx] = 1
            #             ret.append(self.vocab_kge[u])
            #             ret.append(self.vocab_kge[v])
            else:
                pass
        # print('*' * 10)
        # print('self.adj_matrix_1', sum(self.adj_matrix_1))
        return list(set(ret))

    #
    def get_state_rep_kge_2(self):
        ret = []
        self.adj_matrix_2 = np.zeros((len(self.vocab_kge), len(self.vocab_kge)))
        # print('graph_2', self.graph_state_2.edges)
        for u, v in self.graph_state_2.edges:
            u = "_".join(str(u).split()).lower()
            v = "_".join(str(v).split()).lower()
            # print('nodes are => ', u, v)
            if u in self.vocab_kge.keys() and v in self.vocab_kge.keys():
                u_idx = self.vocab_kge[u]
                v_idx = self.vocab_kge[v]
                # print('idx are => ', u_idx, v_idx)
                self.adj_matrix_2[u_idx][v_idx] = 1
                ret.append(self.vocab_kge[u])
                ret.append(self.vocab_kge[v])
            # elif u in self.vocab_kge.keys() and v not in self.vocab_kge.keys():
            #     v = v.split('_')
            #     u_idx = self.vocab_kge[u]
            #     for _v in v:
            #         if _v in self.vocab_kge.keys():
            #             v_idx = self.vocab_kge[_v]
            #             # print('idx are => ', u_idx, v_idx)
            #             self.adj_matrix_2[u_idx][v_idx] = 1
            #             ret.append(self.vocab_kge[u])
            #             ret.append(self.vocab_kge[v])
            # elif v in self.vocab_kge.keys() and u not in self.vocab_kge.keys():
            #     u = u.split('_')
            #     v_idx = self.vocab_kge[v]
            #     for _u in u:
            #         if _u in self.vocab_kge.keys():
            #             u_idx = self.vocab_kge[_u]
            #             # print('idx are => ', u_idx, v_idx)
            #             self.adj_matrix_2[u_idx][v_idx] = 1
            #             ret.append(self.vocab_kge[u])
            #             ret.append(self.vocab_kge[v])
            else:
                pass
        # print('*' * 10)

        return list(set(ret))

    #
    def get_state_rep_kge_3(self):
        ret = []
        self.adj_matrix_3 = np.zeros((len(self.vocab_kge), len(self.vocab_kge)))
        # print('graph_3', self.graph_state_3.edges)
        for u, v in self.graph_state_3.edges:
            u = "_".join(str(u).split()).lower()
            v = "_".join(str(v).split()).lower()
            # print('nodes are => ', u, v)
            if u in self.vocab_kge.keys() and v in self.vocab_kge.keys():
                u_idx = self.vocab_kge[u]
                v_idx = self.vocab_kge[v]
                # print('idx are => ', u_idx, v_idx)
                self.adj_matrix_3[u_idx][v_idx] = 1
                ret.append(self.vocab_kge[u])
                ret.append(self.vocab_kge[v])
            # elif u in self.vocab_kge.keys() and v not in self.vocab_kge.keys():
            #     v = v.split('_')
            #     u_idx = self.vocab_kge[u]
            #     for _v in v:
            #         if _v in self.vocab_kge.keys():
            #             v_idx = self.vocab_kge[_v]
            #             # print('idx are => ', u_idx, v_idx)
            #             self.adj_matrix_3[u_idx][v_idx] = 1
            #             ret.append(self.vocab_kge[u])
            #             ret.append(self.vocab_kge[v])
            # elif v in self.vocab_kge.keys() and u not in self.vocab_kge.keys():
            #     u = u.split('_')
            #     v_idx = self.vocab_kge[v]
            #     for _u in u:
            #         if _u in self.vocab_kge.keys():
            #             u_idx = self.vocab_kge[_u]
            #             # print('idx are => ', u_idx, v_idx)
            #             self.adj_matrix_3[u_idx][v_idx] = 1
            #             ret.append(self.vocab_kge[u])
            #             ret.append(self.vocab_kge[v])
            else:
                pass
        return list(set(ret))

    #
    def get_state_rep_kge_4(self):
        ret = []
        self.adj_matrix_4 = np.zeros((len(self.vocab_kge), len(self.vocab_kge)))
        # print('graph_4', self.graph_state_4.edges)
        for u, v in self.graph_state_4.edges:
            u = "_".join(str(u).split()).lower()
            v = "_".join(str(v).split()).lower()
            # print('nodes are => ', u, v)
            if u in self.vocab_kge.keys() and v in self.vocab_kge.keys():
                u_idx = self.vocab_kge[u]
                v_idx = self.vocab_kge[v]
                # print('idx are => ', u_idx, v_idx)
                self.adj_matrix_4[u_idx][v_idx] = 1
                ret.append(self.vocab_kge[u])
                ret.append(self.vocab_kge[v])
            # elif u in self.vocab_kge.keys() and v not in self.vocab_kge.keys():
            #     v = v.split('_')
            #     u_idx = self.vocab_kge[u]
            #     for _v in v:
            #         if _v in self.vocab_kge.keys():
            #             v_idx = self.vocab_kge[_v]
            #             # print('idx are => ', u_idx, v_idx)
            #             self.adj_matrix_4[u_idx][v_idx] = 1
            #             ret.append(self.vocab_kge[u])
            #             ret.append(self.vocab_kge[v])
            # elif v in self.vocab_kge.keys() and u not in self.vocab_kge.keys():
            #     u = u.split('_')
            #     v_idx = self.vocab_kge[v]
            #     for _u in u:
            #         if _u in self.vocab_kge.keys():
            #             u_idx = self.vocab_kge[_u]
            #             # print('idx are => ', u_idx, v_idx)
            #             self.adj_matrix_4[u_idx][v_idx] = 1
            #             ret.append(self.vocab_kge[u])
            #             ret.append(self.vocab_kge[v])
            else:
                pass

        return list(set(ret))

    # def get_state_rep_kge_1(self):
    #     ret = []
    #     self.adj_matrix_1 = np.zeros((len(self.vocab_kge), len(self.vocab_kge)))
    #     for u, v in self.graph_state_1.edges:
    #         u = '_'.join(str(u).split())
    #         v = '_'.join(str(v).split())
    #         if u not in self.vocab_kge.keys() or v not in self.vocab_kge.keys():
    #             break
    #         u_idx = self.vocab_kge[u]
    #         v_idx = self.vocab_kge[v]
    #         self.adj_matrix_1[u_idx][v_idx] = 1
    #         ret.append(self.vocab_kge[u])
    #         ret.append(self.vocab_kge[v])
    #     return list(set(ret))
    #
    # def get_state_rep_kge_2(self):
    #     ret = []
    #     self.adj_matrix_2 = np.zeros((len(self.vocab_kge), len(self.vocab_kge)))
    #     for u, v in self.graph_state_2.edges:
    #         u = '_'.join(str(u).split())
    #         v = '_'.join(str(v).split())
    #         if u not in self.vocab_kge.keys() or v not in self.vocab_kge.keys():
    #             break
    #         u_idx = self.vocab_kge[u]
    #         v_idx = self.vocab_kge[v]
    #         self.adj_matrix_2[u_idx][v_idx] = 1
    #         ret.append(self.vocab_kge[u])
    #         ret.append(self.vocab_kge[v])
    #     return list(set(ret))
    #
    # def get_state_rep_kge_3(self):
    #     ret = []
    #     self.adj_matrix_3 = np.zeros((len(self.vocab_kge), len(self.vocab_kge)))
    #     for u, v in self.graph_state_3.edges:
    #         u = '_'.join(str(u).split())
    #         v = '_'.join(str(v).split())
    #         if u not in self.vocab_kge.keys() or v not in self.vocab_kge.keys():
    #             break
    #         u_idx = self.vocab_kge[u]
    #         v_idx = self.vocab_kge[v]
    #         self.adj_matrix_3[u_idx][v_idx] = 1
    #         ret.append(self.vocab_kge[u])
    #         ret.append(self.vocab_kge[v])
    #     return list(set(ret))
    #
    # def get_state_rep_kge_4(self):
    #     ret = []
    #     self.adj_matrix_4 = np.zeros((len(self.vocab_kge), len(self.vocab_kge)))
    #     for u, v in self.graph_state_4.edges:
    #         u = '_'.join(str(u).split())
    #         v = '_'.join(str(v).split())
    #         if u not in self.vocab_kge.keys() or v not in self.vocab_kge.keys():
    #             break
    #         u_idx = self.vocab_kge[u]
    #         v_idx = self.vocab_kge[v]
    #         self.adj_matrix_4[u_idx][v_idx] = 1
    #         ret.append(self.vocab_kge[u])
    #         ret.append(self.vocab_kge[v])
    #     return list(set(ret))

    def get_state_rep_kge_5(self):
        ret = []
        self.adj_matrix_5_mask = np.zeros((len(self.vocab_kge), len(self.vocab_kge)))
        for u, v in self.graph_state_5_mask.edges:
            u = "_".join(str(u).split()).lower()
            v = "_".join(str(v).split()).lower()
            if u not in self.vocab_kge.keys() or v not in self.vocab_kge.keys():
                break
            u_idx = self.vocab_kge[u]
            v_idx = self.vocab_kge[v]
            self.adj_matrix_5_mask[u_idx][v_idx] = 1
            ret.append(self.vocab_kge[u])
            ret.append(self.vocab_kge[v])
        return list(set(ret))

    def get_obs_rep(self, *args):
        ret = [self.get_visible_state_rep_drqa(ob) for ob in args]
        return pad_sequences(ret, maxlen=300)

    def get_visible_state_rep_drqa(self, state_description):
        remove = [
            "=",
            "-",
            "'",
            ":",
            "[",
            "]",
            "eos",
            "EOS",
            "SOS",
            "UNK",
            "unk",
            "sos",
            "<",
            ">",
        ]

        for rm in remove:
            state_description = state_description.replace(rm, "")

        return self.sp.encode_as_ids(state_description)

    def get_action_rep_drqa(self, action):

        action_desc_num = 20 * [0]
        action = str(action)

        for i, token in enumerate(action.split()[:20]):
            short_tok = token[: self.max_word_len]
            action_desc_num[i] = (
                self.vocab_act_rev[short_tok] if short_tok in self.vocab_act_rev else 0
            )

        return action_desc_num

    def step(
        self,
        visible_state,
        inventory_state,
        objs,
        prev_action=None,
        cache=None,
        gat=True,
    ):
        ret, ret_cache = self.update_state(
            visible_state, inventory_state, objs, prev_action, cache
        )

        self.pruned_actions_rep = [
            self.get_action_rep_drqa(a) for a in self.vis_pruned_actions
        ]

        inter = (
            self.visible_state
        )  # + "The actions are:" + ",".join(self.vis_pruned_actions) + "."
        self.drqa_input = self.get_visible_state_rep_drqa(inter)

        # Get graph_state_reps
        self.graph_state_rep = self.get_state_rep_kge(), copy.deepcopy(self.adj_matrix)
        self.graph_state_rep_1 = (
            self.get_state_rep_kge_1(),
            copy.deepcopy(self.adj_matrix_1),
        )
        self.graph_state_rep_2 = (
            self.get_state_rep_kge_2(),
            copy.deepcopy(self.adj_matrix_2),
        )
        self.graph_state_rep_3 = (
            self.get_state_rep_kge_3(),
            copy.deepcopy(self.adj_matrix_3),
        )
        self.graph_state_rep_4 = (
            self.get_state_rep_kge_4(),
            copy.deepcopy(self.adj_matrix_4),
        )
        # self.graph_state_rep_5_mask = self.get_state_rep_kge_5(), self.adj_matrix_5_mask

        self.graph_state_rep = self.get_state_rep_kge(), self.adj_matrix

        # adj3 = torch.IntTensor(self.adj_matrix_3)
        # print('represent=>', torch.nonzero(adj3 > 0))

        return ret, ret_cache


def pad_sequences(sequences, maxlen=None, dtype="int32", value=0.0):
    """
    Partially borrowed from Keras
    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        value: float, value to pad the sequences to the desired value.
    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)
    """
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break
    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        # pre truncating
        trunc = s[-maxlen:]
        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError(
                "Shape of sample %s of sequence at position %s is different from expected shape %s"
                % (trunc.shape[1:], idx, sample_shape)
            )
        # post padding
        x[idx, : len(trunc)] = trunc
    return x