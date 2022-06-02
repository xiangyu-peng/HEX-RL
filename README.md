### :thinking: Explainable RL
#### How to use [QBert](https://github.com/rajammanabrolu/Q-BERT)?
* Follow the [README](https://github.com/rajammanabrolu/Q-BERT-internal/blob/explain_RL/qbert/README.md).
* Issues may happen:
    * `ModuleNotFoundError: No module named 'flask'`: pls go [there](https://stackoverflow.com/questions/18776745/gunicorn-with-flask-using-wrong-python).
    * `RuntimeError: cuda runtime error (804) : forward compatibility was attempted on non supported HW at /pytorch/aten/src/THC/THCGeneral.cpp:50`: pls follow [this](https://stackoverflow.com/questions/43022843/nvidia-nvml-driver-library-version-mismatch).
 
#### How to use [SHA-KG](https://github.com/YunqiuXu/SHA-KG)?
* Follow KG-A2C's [README](https://github.com/rajammanabrolu/KG-A2C).
* Then Follow SHA-KG's [README](https://github.com/YunqiuXu/SHA-KG).
* Remember to download stanford CoreNLP by 

```ruby
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip
unzip stanford-corenlp-full-2018-10-05.zip
cd stanford-corenlp-full-2018-10-05/ && java -mx8g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
```      

* If you have issues with `redis`, modify the port number for redis (from default 6381 to **6379**) and corenlp (from default 9010 to **9000**) in `env.py`, `openie.py` and `vec_env.py`

#### :collision: Replace R-GCN in QBert with SHA-KG
* Use [tmux](https://www.hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/)
* Open one terminal:

```ruby
cd qbert/extraction && gunicorn --workers 4 --bind 0.0.0.0:5000 wsgi:app
redis-server
```

* Open another terminal:
```ruby
cd qbert && python train.py --training_type base --reward_type game_only  --subKG_type QBert
```
```ruby
nohup python train.py --training_type chained --reward_type game_and_IM  --subKG_type QBert --batch_size 2 --seed 0 --preload_weights Q-BERT/qbert/logs/qbert.pt --eval_mode --graph_dropout 0 --mask_dropout 0 --dropout_ratio 0
```

#### :eye_speech_bubble: Features
* `--subKG_type`: What kind of subgraph you want to use. There are 3 choices, 'Full', 'SHA', 'QBert'.
    * 'Full': 4 subgraphs are all full graph_state.
    * 'QBert':
        1. __ 'is' __ (Attr of objects)
        2. 'you' 'have' __
        3. __ 'in' __
        4. others (direction)
    * 'SHA':
        1. room connectivity (history included)
        2. what's in current room
        3. your inventory
        4. remove you related nodes (history included)

* `--eval_mode`: Whether turning off the training and evaluation the pre-trained model
    * bool. True or False
    * use `--preload_weights` at the same time.
    
* `--random_action`: Whether to use random valid actions instead of QBERT actions.
    * bool. True or False


#### :sweat_drops: Things need to do
- [x] Increase the threshold of `animate`, because we see `animate` and `absent` every time. `--extraction 0.5`
- [x] Look at the ![formula](https://render.githubusercontent.com/render/math?math=\color{red}\alpha_{\text{high}})
    - [x] The encoding dimension is `4 * 100`. Mask out paddings and change the coefficent matrix as the same dimension.
    - [x] Find the index of the highest ![formula](https://render.githubusercontent.com/render/math?math=\color{red}\alpha_{\text{high}})
    - [x] sp can be find [here](https://github.com/google/sentencepiece). Use `self.sp.decode()` to decode the important words or phrases
- [x] Take all the log. 
    - [x] A txt file with (1) observations, (2) action, whether this is bottleneck
    - [x] KG - a file with step name
    - [x] Another text file for the explanation of this step
    - [x] top_k_attention value. - a file with step name
- [x] Run the SHA-KG-QBERT until score >= 30.
- [x] Create flags - Use 4 full graph to replace 4 SHA-KG and run until reward = 30
- [x] Create flags - Design new 4 subgraphs
    * __ 'is' __ (Attr of objects)
    * 'you' 'have' __
    * __ 'in' __
    * others (direction)
- [x] Play around with the [attribute confidence](https://github.com/rajammanabrolu/Q-BERT-internal/blob/0845637eb1f5b56155798cbc30547459d422dbab/qbert/extraction/kg_extraction.py#L95) threshold before you start the runs to see if we can get more than animate and absent.
    * Tune `--extraction`
- [ ] Merge the [ground_truth](https://github.com/rajammanabrolu/Q-BERT-internal/blob/master/kga2c/train.py#L124) KG. 

#### Debug Tricks
1. graph_dropout to .5 and mask_dropout to .5 in `train.py`.
2. The score should reach 5 in 10,000 steps.

 python train.py --training_type chained --reward_type game_and_IM  --subKG_type QBert --batch_size 2 --seed 0 --preload_weights Q-BERT/qbert/logs/qbert.pt --eval_mode True --graph_dropout 0 --mask_dropout 0 --dropout_ratio 0 --step 50 --seed 4 --random_action True
