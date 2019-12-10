import os

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

create_dir('data')
create_dir('data/raw')
create_dir('data/working')
create_dir('output')
