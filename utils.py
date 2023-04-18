from omegaconf import OmegaConf
from pathlib import Path


def parse_config_file(parser):
    args = parser.parse_args()
    # Reading the YAML config file
    params = OmegaConf.load(Path(Path(__file__).parent.resolve() / 'configs' / args.config))
    # Defining the root directory of the project
    params.root_dir = str(Path(__file__).parent.resolve())
    # Defining the path to the dataset if not specified in the config file
    if params.dataset_path is None:
        params.dataset_path = str(Path(Path.home() / 'datasets' / params.dataset_name))

    return params
