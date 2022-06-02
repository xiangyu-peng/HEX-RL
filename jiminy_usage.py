import sys
sys.path.insert(0, 'Q-BERT/qbert/jiminy-cricket')
from annotated_env import AnnotatedEnv

game_name = 'zork1'  # change to desired game
env = AnnotatedEnv(game_folder_path='Q-BERT/qbert/jiminy-cricket/annotated_games/{}'.format(game_name))
print(env.reset())
print(env.step(action='open mailbox'))
print(env.step(action='read leaflet'))
print(env.step(action='go south'))
print(env.step(action='go east'))
print(env.step(action='open window'))
print(env.step(action='enter house'))