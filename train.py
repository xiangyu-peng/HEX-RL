# ps ax|grep gunicorn
import os, sys
from intrinsic_qbert import QBERTTrainer
import argparse
from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
)

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
}

game = 'zork1'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--game_name",
        type=str,
        default=game,
        help="GAme we play",
    )

    # GoExplore params
    parser.add_argument(
        "--resolution",
        "--res",
        type=float,
        default=16,
        help="Length of the side of a grid cell.",
    )
    parser.add_argument(
        "--use_scores",
        dest="use_objects",
        action="store_false",
        help="Use scores in the cell description. Otherwise objects will be used.",
    )
    parser.add_argument(
        "--repeat_action",
        "--ra",
        type=int,
        default=20,
        help="The average number of times that actions will be repeated in the exploration phase.",
    )
    parser.add_argument(
        "--explore_steps",
        type=int,
        default=100,
        help="Maximum number of steps in the explore phase.",
    )
    parser.add_argument(
        "--ignore_death",
        type=int,
        default=1,
        help="Number of steps immediately before death to ignore.",
    )
    parser.add_argument(
        "--base_path",
        "-p",
        type=str,
        default="./results/",
        help="Folder in which to store results",
    )
    parser.add_argument(
        "--path_postfix",
        "--pf",
        type=str,
        default="",
        help="String appended to the base path.",
    )
    parser.add_argument(
        "--seed_path",
        type=str,
        default=None,
        help="Path from which to load existing results.",
    )
    parser.add_argument(
        "--x_repeat",
        type=int,
        default=2,
        help="How much to duplicate pixels along the x direction. 2 is closer to how the games were meant to be played, but 1 is the original emulator resolution. NOTE: affects the behavior of GoExplore.",
    )
    parser.add_argument(
        "--seen_weight",
        "--sw",
        type=float,
        default=0.0,
        help='The weight of the "seen" attribute in cell selection.',
    )
    parser.add_argument(
        "--seen_power",
        "--sp",
        type=float,
        default=0.5,
        help='The power of the "seen" attribute in cell selection.',
    )
    parser.add_argument(
        "--chosen_weight",
        "--cw",
        type=float,
        default=1.0,
        help='The weight of the "chosen" attribute in cell selection.',
    )
    parser.add_argument(
        "--chosen_power",
        "--cp",
        type=float,
        default=0.5,
        help='The power of the "chosen" attribute in cell selection.',
    )
    parser.add_argument(
        "--chosen_since_new_weight",
        "--csnw",
        type=float,
        default=1.0,
        help='The weight of the "chosen since new" attribute in cell selection.',
    )
    parser.add_argument(
        "--chosen_since_new_power",
        "--csnp",
        type=float,
        default=1.0,
        help='The power of the "chosen since new" attribute in cell selection.',
    )
    parser.add_argument(
        "--action_weight",
        "--aw",
        type=float,
        default=0.0,
        help='The weight of the "action" attribute in cell selection.',
    )
    parser.add_argument(
        "--action_power",
        "--ap",
        type=float,
        default=0.5,
        help='The power of the "action" attribute in cell selection.',
    )
    parser.add_argument(
        "--horiz_weight",
        "--hw",
        type=float,
        default=1.0,
        help="Weight of not having one of the two possible horizontal neighbors.",
    )
    parser.add_argument(
        "--vert_weight",
        "--vw",
        type=float,
        default=0.0,
        help="Weight of not having one of the two possible vertical neighbors.",
    )
    parser.add_argument(
        "--low_score_weight",
        type=float,
        default=0.0,
        help="Weight of not having a neighbor with a lower score/object number.",
    )
    parser.add_argument(
        "--high_score_weight",
        type=float,
        default=0.5,
        help="Weight of not having a neighbor with a higher score/object number.",
    )
    parser.add_argument(
        "--end_on_death",
        dest="end_on_death",
        action="store_true",
        help="End episode on death.",
    )
    parser.add_argument(
        "--low_level_weight",
        type=float,
        default=0.1,
        help="Weight of cells in levels lower than the current max. If this is non-zero, lower levels will keep getting optimized, potentially leading to better solutions overall. Setting this to greater than 1 is possible but nonsensical since it means putting a larger weight on low levels than higher levels.",
    )
    parser.add_argument(
        "--max_game_steps",
        type=int,
        default=None,
        help="Maximum number of GAME frames.",
    )
    parser.add_argument(
        "--max_compute_steps",
        "--mcs",
        type=int,
        default=1000000000,
        help="Maximum number of COMPUTE frames.",
    )
    parser.add_argument(
        "--max_iterations", type=int, default=None, help="Maximum number of iterations."
    )
    parser.add_argument(
        "--max_hours",
        "--mh",
        type=float,
        default=500,
        help="Maximum number of hours to run this for.",
    )
    parser.add_argument(
        "--checkpoint_game",
        type=int,
        default=20_000_000_000_000,
        help="Save a checkpoint every this many GAME frames (note: recommmended to ignore, since this grows very fast at the end).",
    )
    parser.add_argument(
        "--checkpoint_compute",
        type=int,
        default=1_000_000,
        help="Save a checkpoint every this many COMPUTE frames.",
    )
    parser.add_argument(
        "--pictures",
        dest="save_pictures",
        action="store_true",
        help="Save pictures of the pyramid every checkpoint (uses more space).",
    )
    parser.add_argument(
        "--prob_pictures",
        "--pp",
        dest="save_prob_pictures",
        action="store_true",
        help="Save pictures of showing probabilities.",
    )
    parser.add_argument(
        "--item_pictures",
        "--ip",
        dest="save_item_pictures",
        action="store_true",
        help="Save pictures of showing items collected.",
    )
    parser.add_argument(
        "--keep_checkpoints",
        dest="clear_old_checkpoints",
        action="store_false",
        help="Keep all checkpoints in large format. This isn't necessary for view folder to work. Uses a lot of space.",
    )
    parser.add_argument(
        "--keep_prob_pictures",
        "--kpp",
        dest="keep_prob_pictures",
        action="store_true",
        help="Keep old pictures showing probabilities.",
    )
    parser.add_argument(
        "--keep_item_pictures",
        "--kip",
        dest="keep_item_pictures",
        action="store_true",
        help="Keep old pictures showing items collected.",
    )
    parser.add_argument(
        "--no_warn_delete",
        dest="warn_delete",
        action="store_false",
        help="Do not warn before deleting the existing directory, if any.",
    )
    parser.add_argument(
        "--game",
        "-g",
        type=str,
        default="zork",
        help="Determines the game to which apply goexplore.",
    )
    parser.add_argument(
        "--objects_from_ram",
        dest="objects_from_pixels",
        action="store_false",
        help="Get the objects from RAM instead of pixels.",
    )
    parser.add_argument(
        "--all_objects",
        dest="only_keys",
        action="store_false",
        help="Use all objects in the state instead of just the keys",
    )
    parser.add_argument(
        "--remember_rooms",
        dest="remember_rooms",
        action="store_true",
        help="Remember which room the objects picked up came from. Makes it easier to solve the game (because the state encodes the location of the remaining keys anymore), but takes more time/memory space, which in practice makes it worse quite often. Using this is better if running with --no_optimize_score",
    )
    parser.add_argument(
        "--no_optimize_score",
        dest="optimize_score",
        action="store_false",
        help='Don\'t optimize for score (only speed). Will use fewer "game frames" and come up with faster trajectories with lower scores. If not combined with --remember_rooms and --objects_from_ram is not enabled, things should run much slower.',
    )
    parser.add_argument(
        "--prob_override",
        type=float,
        default=0.0,
        help="Probability that the newly found cells will randomly replace the current cell.",
    )
    parser.add_argument(
        "--resize_x",
        "--rx",
        type=int,
        default=11,
        help="What to resize the pixels to in the x direction for use as a state.",
    )
    parser.add_argument(
        "--resize_y",
        "--ry",
        type=int,
        default=8,
        help="What to resize the pixels to in the y direction for use as a state.",
    )
    parser.add_argument(
        "--state_is_pixels",
        "--pix",
        dest="state_is_pixels",
        action="store_true",
        help="If this is on, the state will be resized pixels, not human prior.",
    )
    parser.add_argument(
        "--max_pix_value",
        "--mpv",
        type=int,
        default=8,
        help="The range of pixel values when resizing will be rescaled to from 0 to this value. Lower means fewer possible states in states_is_pixels.",
    )
    parser.add_argument(
        "--n_cpus", type=int, default=None, help="Number of worker threads to spawn"
    )
    parser.add_argument(
        "--go_batch_size", type=int, default=1, help="Number of worker threads to spawn"
    )
    parser.add_argument(
        "--pool_class",
        type=str,
        default="py",
        help="The multiprocessing pool class (py or torch).",
    )
    parser.add_argument(
        "--start_method", type=str, default="fork", help="The process start method."
    )
    parser.add_argument(
        "--reset_pool",
        dest="reset_pool",
        action="store_true",
        help="The pool should be reset every 100 iterations.",
    )
    parser.add_argument(
        "--reset_cell_on_update",
        "--rcou",
        dest="reset_cell_on_update",
        action="store_false",
        help="Reset the times-chosen and times-chosen-since when a cell is updated.",
    )
    parser.add_argument(
        "--profile",
        dest="profile",
        action="store_true",
        help="Whether or not to enable a profiler.",
    )

    # Base QBERT Params
    parser.add_argument(
        "--output_dir", default="/Q-BERT/qbert/logs/" + game + '/'
    )
    parser.add_argument("--spm_file", default="./spm_models/unigram_8k.model")
    parser.add_argument("--tsv_file", default="../data/" + game + "_entity2id.tsv")
    parser.add_argument("--attr_file", default="attrs/" + game + "_attr.txt")
    parser.add_argument(
        "--rom_file_path",
        default="/Q-BERT/z-machine-games-master/jericho-game-suite/" + game + ".z5",
    )
    parser.add_argument("--checkpoint_path", default="./models/checkpoints/")
    parser.add_argument("--redis_db_path", default="")
    parser.add_argument(
        "--openie_path",
        default="/Q-BERT/stanford-corenlp-full-2018-10-05",
    )
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--lr", default=0.003, type=float)
    parser.add_argument("--gamma", default=0.9, type=float)
    parser.add_argument("--embedding_size", default=50, type=int)
    parser.add_argument("--hidden_size", default=100, type=int)
    parser.add_argument("--padding_idx", default=0, type=int)
    parser.add_argument("--gat_emb_size", default=25, type=int)
    parser.add_argument("--dropout_ratio", default=0.2, type=float)
    parser.add_argument("--bindings", default=game)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--steps", default=20000000, type=int)
    parser.add_argument("--reset_steps", default=100, type=int)
    parser.add_argument("--stuck_steps", default=100, type=int)
    parser.add_argument("--trial", default="base")
    parser.add_argument("--loss", default="value_policy_entropy")
    parser.add_argument("--graph_dropout", default=0.2, type=float)
    parser.add_argument("--mask_dropout", default=0.1, type=float)
    parser.add_argument("--k_object", default=1, type=int)
    parser.add_argument("--g_val", default=False, type=bool)
    parser.add_argument("--entropy_coeff", default=0.03, type=float)
    parser.add_argument("--clip", default=40, type=int)
    parser.add_argument("--bptt", default=8, type=int)
    parser.add_argument("--value_coeff", default=9, type=float)
    parser.add_argument("--template_coeff", default=3, type=float)
    parser.add_argument("--object_coeff", default=9, type=float)
    parser.add_argument("--recurrent", default=True, type=bool)
    parser.add_argument("--checkpoint_interval", default=100, type=int)
    parser.add_argument("--no-gat", dest="gat", action="store_false")
    parser.add_argument(
        "--masking",
        default="kg",
        choices=["kg", "interactive", "none"],
        help="Type of object masking applied",
    )  #

    parser.add_argument("--patience", default=3000, type=int)
    parser.add_argument("--buffer_size", default=40, type=int)
    parser.add_argument("--epsilon", default=1e-2, type=float)
    parser.add_argument("--kg_diff_threshold", default=6, type=int)
    parser.add_argument("--kg_diff_batch_percentage", default=0.4, type=float)
    parser.add_argument("--intrinsic_motivation_factor", default=2, type=float)
    parser.add_argument(
        "--patience_valid_only",
        default=True,
        type=bool,
        help="only counting valid actions",
    )
    parser.add_argument("--patience_batch_factor", default=0.75, type=float, help="1")
    parser.add_argument("--clear_kg_on_reset", default=False, type=bool)
    parser.add_argument("--chained_logger", default="logs/chained.log")
    parser.add_argument("--goexplore_logger", default="logs/goexplore.log")

    parser.add_argument(
        "--reward_type",
        default="game_only",
        choices=["game_only", "IM_only", "game_and_IM"],
    )
    parser.add_argument(
        "--training_type", default="base", choices=["base", "chained", "goexplore"]
    )
    parser.add_argument("--extraction", default=0.2, type=float)
    parser.set_defaults(gat=True)

    ##### BBB #####
    parser.add_argument(
        "--subKG_type",
        default="QBert",
        type=str,
        choices=["Full", "SHA", "QBert"],
        help="What kind of sub graph we build.",
    )
    parser.add_argument(
        "--eval_mode",
        action='store_true',
        help="Whether turning off the training and evaluation the pre-trained model",
    )
    parser.add_argument(
        "--early_stop_score",
        default=60,
        type=int,
        help="The score limit of early stopping",
    )
    parser.add_argument(
        "--early_stop_score_count",
        default=0,
        type=int,
        help="Max score less than limit can be accepted as early stopping",
    )
    parser.add_argument("--preload_weights", default="")
    #################
    # GPT-2
    parser.add_argument(
        "--model_type",
        default="gpt2",
        type=str,
        required=False,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default="gpt2",
        type=str,
        required=False,
        help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(MODEL_CLASSES.keys()),
    )

    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument(
        "--stop_token",
        type=str,
        default=None,
        help="Token at which text generation is stopped",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="primarily useful for CTRL model; in that case, use 1.2",
    )
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)

    parser.add_argument(
        "--prefix", type=str, default="", help="Text added prior to input."
    )
    parser.add_argument(
        "--padding_text",
        type=str,
        default="",
        help="Deprecated, the use of `--prefix` is preferred.",
    )
    parser.add_argument(
        "--xlm_language",
        type=str,
        default="",
        help="Optional language when used with the XLM model.",
    )

    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="The number of samples to generate.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )

    parser.add_argument(
        "--argmax_sig",
        action="store_true",
        help="Whether to use argmax actions",
    )
    parser.add_argument(
        "--alpha_gat",
        type=float,
        default=0.2,
        help="Whether to use argmax actions",
    )
    parser.add_argument("--device_id", type=int, default=0, help="gpu id")
    parser.add_argument("--random_action", type=bool, default=False, help="True means random actions")
    args = parser.parse_args()
    params = vars(args)
    return params, args


if __name__ == "__main__":
    params, args = parse_args()
    print(params)

    # create dir
    if os.path.isdir(params['output_dir']):
        pass
    else:
        os.mkdir(params['output_dir'])


    if params["training_type"] != "goexplore" and not params["eval_mode"]:
        print("===eval mode is off====")
        trainer = QBERTTrainer(params, args)
        trainer.train(params["steps"])
    elif params["eval_mode"]:
        print("===eval mode is on====")
        trainer = QBERTTrainer(params, args)
        trainer.test(params["steps"])
    else:
        from goexplore_py import goexplore_main

        sys.path.append(".")
        params["batch_size"] = 1
        goexplore_main.main(args, params)
