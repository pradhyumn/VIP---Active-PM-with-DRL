# Import necessary libraries
import os
import shutil
import logging
import numpy as np
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pgportfolio import constants
# Import necessary components for data management, backtesting and training from pgportfolio
from pgportfolio.marketdata.coin_data_manager import \
    coin_data_manager_init_helper
from pgportfolio.trade.backtest import BackTest
from pgportfolio.nnagent.tradertrainer import TraderTrainer
# Import utility functions from pgportfolio
from pgportfolio.utils.config import load_config, save_config
from pgportfolio.utils import plot
import sqlite3

# Define a function to build a command line argument parser
def build_parser():
    parser = ArgumentParser()
    # Add arguments to the parser for mode, proxy, offline, algos, labels, format, device, working directory, and config
    # Each argument has a default value and a help text
    # User can override these defaults when running the script
    parser.add_argument("--mode", dest="mode",
                        help="train, download_data, save_test_data, "
                             "backtest, plot, table",
                        metavar="MODE", default="train")
    parser.add_argument("--proxy",
                        help='socks proxy',
                        dest="proxy", default="")
    parser.add_argument("--offline", dest="offline", action="store_true",
                        help="Use local database data if set.")
    parser.add_argument("--algos",
                        help='algo names, seperated by ","',
                        dest="algos")
    parser.add_argument("--labels", dest="labels",
                        help="names that will shown in the figure caption "
                             "or table header")
    parser.add_argument("--format", dest="format", default="raw",
                        help="format of the table printed")
    parser.add_argument("--device", dest="device", default="cpu",
                        help="device to be used to train, use number 0 to "
                             "indicate gpu device like cuda:0")
    parser.add_argument("--working_dir", dest="working_dir",
                        default=constants.ROOT_DIR,
                        help="Working directory, by default it is project root")
    parser.add_argument("--config", dest="config",
                        default=constants.CONFIG_FILE,
                        help="Config file, by default it is "
                             "config.json")
    return parser

# Define a function to create necessary directories for the project
def prepare_directories(root):
    # Check if each directory exists, if not create it
    # The directories include root, database, log, model and result directories
    # The log directory further contains a subdirectory for tensorboard logs
    # This function ensures that the project has the necessary directory structure
    if not os.path.exists(root):
        os.makedirs(root)
    if not os.path.exists(root + "/database"):
        os.makedirs(root + "/database")
    if not os.path.exists(root + "/log"):
        os.makedirs(root + "/log")
        os.makedirs(root + "/log/tensorboard_log")
    if not os.path.exists(root + "/model"):
        os.makedirs(root + "/model")
    if not os.path.exists(root + "/result"):
        os.makedirs(root + "/result")

# Main function to start execution
def main():
    # Get the command line arguments
    parser = build_parser()
    options = parser.parse_args()
    # Prepare necessary directories
    prepare_directories(options.working_dir)
    # Basic configuration for logging
    logging.basicConfig(level=logging.INFO)

    # If a proxy is specified, use it for network requests
    if options.proxy != "":
        # monkey patching
        addr, port = options.proxy.split(":")
        constants.PROXY_ADDR = addr
        constants.PROXY_PORT = int(port)
    # Log a message if running in offline mode
    if options.offline:
        logging.info("Note: in offline mode.")

    # If mode is "train", start training the model
    if options.mode == "train":
        # delete old models
        shutil.rmtree(options.working_dir + "/model")
        # Load the configuration and save it
        config = load_config(options.config)
        save_config(config, options.working_dir + "/config.json")
        print("--------------------")
        print('{test_portfolio_value:.2f}')
        print('{epoch:02d}')
        # Create a model checkpoint callback
        checkpoint_callback = ModelCheckpoint(\
            dirpath=options.working_dir + "/model",
            filename="{epoch:02d}-{test_portfolio_value:.2f}",
            save_top_k=1,
            monitor="test_portfolio_value", mode="max",
            every_n_epochs=None, verbose=True
        )
        # Create an early stopping callback
        early_stopping = EarlyStopping(
            monitor="test_portfolio_value", mode="max"
        )
        # Create a TensorBoard logger
        t_logger = TensorBoardLogger(
            options.working_dir + "/log/tensorboard_log"
        )
        # Create a PyTorch Lightning trainer and start training
        trainer = pl.Trainer( 
            accelerator="cpu",        
            callbacks=[checkpoint_callback],
            logger=[t_logger],
            limit_train_batches=1000, #No of training steps in each epoch
            max_steps=config["training"]["steps"]
        )
        # Initialize and train the model
        model = TraderTrainer(config,
                              online=not options.offline,
                              db_directory=options.working_dir + "/database")
        print("Line 107.......................",options.working_dir + "/database")
        trainer.fit(model)

    # If mode is "download_data", download the required data
    elif options.mode == "download_data":
        config = load_config(options.config)
        coin_data_manager_init_helper(
            config, download=True,
            online=not options.offline,
            db_directory=options.working_dir + "/database"
        )

    # If mode is "backtest", perform backtesting on the specified algorithms
    elif options.mode == "backtest":
        if options.algos is None:
            raise ValueError("Algorithms not set.")
        config = load_config(options.config)
        save_config(config, options.working_dir + "/config.json")
        algos = options.algos.split(",")
        backtests = [BackTest(config,
                              agent_algorithm=algo,
                              online=not options.offline,
                              verbose=True,
                              model_directory=options.working_dir + "/model",
                              db_directory=options.working_dir + "/database")
                     for algo in algos]
        for b in backtests:
            b.trade()

    # If mode is "save_test_data", export the test data
    elif options.mode == "save_test_data":
        # This is used to export the test data
        config = load_config(options.config)
        backtest = BackTest(config, agent_algorithm="not_used",
                            online=not options.offline,
                            model_directory=options.working_dir + "/model",
                            db_directory=options.working_dir + "/database")
        with open(options.working_dir + "/test_data.csv", 'wb') as f:
            np.savetxt(f, backtest.test_data.T, delimiter=",")

    # If mode is "plot", plot backtest results for given algorithms
    elif options.mode == "plot":
        if options.algos is None:
            raise ValueError("Algorithms not set.")
        config = load_config(options.config)
        algos = options.algos.split(",")
        # If labels are provided, use them for plots. Otherwise, use algo names as labels
        if options.labels:
            labels = options.labels.replace("_", " ")
            labels = labels.split(",")
        else:
            labels = algos
        plot.plot_backtest(config, algos, labels,
                           online=not options.offline,
                           working_directory=options.working_dir,
                           model_directory=options.working_dir + "/model",
                           db_directory=options.working_dir + "/database")
        
    # If mode is "table", create a table of backtest results for given algorithms
    elif options.mode == "table":
        if options.algos is None:
            raise ValueError("Algorithms not set.")
        config = load_config(options.config)
        algos = options.algos.split(",")
        # If labels are provided, use them for table headers. Otherwise, use algo names as labels
        if options.labels:
            labels = options.labels.replace("_", " ")
            labels = labels.split(",")
        else:
            labels = algos
        plot.table_backtest(config, algos, labels,
                            format=options.format,
                            online=not options.offline,
                            working_directory=options.working_dir,
                            model_directory=options.working_dir + "/model",
                            db_directory=options.working_dir + "/database")


if __name__ == "__main__":
    main()
