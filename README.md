## HEX-RL
This code accompanies the paper [Inherently Explainable Reinforcement Learning in Natural Language](https://arxiv.org/abs/2112.08907).

### :thinking: Explainable RL

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


#### Debug Tricks
1. graph_dropout to .5 and mask_dropout to .5 in `train.py`.
2. The score should reach 5 in 10,000 steps.
