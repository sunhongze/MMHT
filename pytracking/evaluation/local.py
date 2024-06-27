from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.eotb_path = '/FE108'   # This is the path of FE108 dataset
    settings.network_path = './pytracking/networks/'    # Where tracking networks are stored.
    settings.result_plot_path = './pytracking/result_plots/'
    settings.results_path = './pytracking/tracking_results/'    # Where to store tracking results
    settings.segmentation_path = './pytracking/segmentation_results/'
    return settings

