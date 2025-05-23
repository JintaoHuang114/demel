from omegaconf import OmegaConf


def setup_parser():
    args = OmegaConf.load("../config/wikimel.yaml")
    return args
