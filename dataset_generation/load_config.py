#temporary jank
def load_config(globals):
    with open('audioset_tagging_cnn/utils/config.py') as f:
        config_code = f.read()
        exec(config_code, globals)