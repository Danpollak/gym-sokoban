from .sokoban_env import SokobanEnv
from .render_utils import room_to_rgb
import os
from os import listdir
from os.path import isfile, join
import requests
import zipfile
from tqdm import tqdm
import random
import numpy as np

class BoxobanEnv(SokobanEnv):
    num_boxes = 4
    dim_room=(10, 10)

    def __init__(self,
             max_steps=120,
             difficulty='unfiltered', split='train'):
        self.difficulty = difficulty
        self.split = split
        self.verbose = False
        super(BoxobanEnv, self).__init__(self.dim_room, max_steps, self.num_boxes, None)
        

    def reset(self):
        self.cache_path = '.sokoban_cache'
        self.train_data_dir = os.path.join(self.cache_path, 'boxoban-levels-master', self.difficulty, self.split)

        if not os.path.exists(self.cache_path):
           
            url = "https://github.com/deepmind/boxoban-levels/archive/master.zip"
            
            if self.verbose:
                print('Boxoban: Pregenerated levels not downloaded.')
                print('Starting download from "{}"'.format(url))

            response = requests.get(url, stream=True)

            if response.status_code != 200:
                raise "Could not download levels from {}. If this problem occurs consistantly please report the bug under https://github.com/mpSchrader/gym-sokoban/issues. ".format(url)

            os.makedirs(self.cache_path)
            path_to_zip_file = os.path.join(self.cache_path, 'boxoban_levels-master.zip')
            with open(path_to_zip_file, 'wb') as handle:
                for data in tqdm(response.iter_content()):
                    handle.write(data)

            zip_ref = zipfile.ZipFile(path_to_zip_file, 'r')
            zip_ref.extractall(self.cache_path)
            zip_ref.close()
        
        self.select_room()

        self.num_env_steps = 0
        self.reward_last = 0
        self.boxes_on_target = 0

        starting_observation = room_to_rgb(self.room_state, self.room_fixed)

        return starting_observation

    def select_room(self):
        
        generated_files = [f for f in listdir(self.train_data_dir) if isfile(join(self.train_data_dir, f))]
        source_file = join(self.train_data_dir, random.choice(generated_files))

        maps = []
        current_map = []
        
        with open(source_file, 'r') as sf:
            for line in sf.readlines():
                if ';' in line and current_map:
                    maps.append(current_map)
                    current_map = []
                if '#' == line[0]:
                    current_map.append(line.strip())
        
        maps.append(current_map)

        selected_map = random.choice(maps)

        if self.verbose:
            print('Selected Level from File "{}"'.format(source_file))

        self.room_fixed, self.room_state, self.box_mapping = self.generate_room(selected_map)


    def room_f_code(e):
        if e == '#':
            return 0

        elif e == '@':
            return 1

        elif e == '$':
            return 1

        elif e == '.':
            return 2

        return 1

    def room_s_code(e):
        if e == '#':
            return 0

        elif e == '@':
            return 5

        elif e == '$':
            return 4

        elif e == '.':
            return 2

        return 1
        

    def generate_room(self, select_map):
        room_fixed = []
        room_state = []

        room_fixed = [room_f_code(e) for e in row for row in select_map]
        room_state = [room_s_code(e) for e in row for row in select_map]

        # used for replay in room generation, unused here because pre-generated levels
        box_mapping = {}

        return np.array(room_fixed), np.array(room_state), box_mapping


