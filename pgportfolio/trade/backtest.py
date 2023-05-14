import os
import re
import logging
import numpy as np
from pgportfolio.nnagent.rollingtrainer import RollingTrainer
from pgportfolio.nnagent.replay_buffer import buffer_init_helper
from pgportfolio.tdagent.algorithms import \
    crp, ons, olmar, up, anticor1, pamr,\
    best, bk, cwmr_std, eg, sp, ubah, \
    wmamr, bcrp, cornk, m0, rmr


# the dictionary of name of algorithms mapping to the constructor of tdagents
ALGOS = {"crp": crp.CRP, "ons": ons.ONS, "olmar": olmar.OLMAR, "up": up.UP,
         "anticor": anticor1.ANTICOR1, "pamr": pamr.PAMR,
         "best": best.BEST, "bk": bk.BK, "bcrp": bcrp.BCRP,
         "corn": cornk.CORNK, "m0": m0.M0, "rmr": rmr.RMR,
         "cwmr": cwmr_std.CWMR_STD, "eg": eg.EG, "sp": sp.SP, "ubah": ubah.UBAH,
         "wmamr": wmamr.WMAMR}


traditional_data_cache = None


class BackTest:
    def __init__(self, config,
                 initial_BTC=1.0,
                 agent_algorithm="nn",
                 online=True,
                 verbose=False,
                 model_directory=None,
                 db_directory=None):
        """
        Args:
            config: Config dictionary.
            initial_BTC: Initial BTC amount.
            agent_algorithm: "nn" for nnagent, or anything in tdagent,
            or "not_used" for pure data extraction with no agent.
        """

        # Cache for traditional data
        global traditional_data_cache
        # Initialize some class variables
        self._steps = 0
        self._agent_alg = agent_algorithm
        self._verbose = verbose
        # Log the agent algorithm being used
        logging.info("Creating test agent {}".format(agent_algorithm))
        # If the agent algorithm is neural network (nn)
        if agent_algorithm == "nn":
            # Find the latest checkpoint in the model directory
            ckpt = self._find_latest_checkpoint(model_directory)
            # Log the checkpoint being loaded
            logging.info("Loading checkpoint {} for nn agent".format(ckpt))
            # Load the model from the checkpoint
            self._rolling_trainer = RollingTrainer.load_from_checkpoint(
                model_directory + "/" + ckpt, map_location="cpu",
                config=config,
                online=online,
                db_directory=db_directory
            )
            # Get the list of coins and the agent
            self._coin_name_list = self._rolling_trainer.coins
            self._agent = self._rolling_trainer
            # Get the test set
            test_set = self._rolling_trainer.test_set
        # If the agent algorithm is one of the traditional methods or not used
        elif agent_algorithm in ALGOS or agent_algorithm == "not_used":
            config = config.copy()
            config["input"]["feature_number"] = 1
            # If the traditional data cache is empty, initialize it
            if traditional_data_cache is None:
                cdm, buffer = buffer_init_helper(
                    config, "cpu", online=online, db_directory=db_directory
                )
                test_set = buffer.get_test_set()
                traditional_data_cache = {"test_set": test_set,
                                          "coin_name_list": cdm.coins}
                self._coin_name_list = cdm.coins
            else:
                # If the cache is not empty, use the cached data
                test_set = traditional_data_cache["test_set"]
                self._coin_name_list = traditional_data_cache["coin_name_list"]
            # If the agent algorithm is not "not_used", initialize the agent
            if agent_algorithm != "not_used":
                self._agent = ALGOS[agent_algorithm]()
        else:
            # If the agent algorithm is not supported, raise an error
            raise ValueError('The algorithm name "{}" is not supported. '
                             'Supported algorithms are {}'
                             .format(agent_algorithm, str(list(ALGOS.keys()))))

        # Convert the test set to numpy arrays
        self._test_set_X = test_set["X"].cpu().numpy()
        self._test_set_y = test_set["y"].cpu().numpy()
        # Initialize other variables for the backtest
        self._test_set_length = self._test_set_X.shape[0]
        self._test_pv = 1.0
        self._test_pc_vector = []

        # the total assets is calculated with BTC
        self._total_capital = initial_BTC
        self._coin_number = config["input"]["coin_number"]
        self._commission_rate = config["trading"]["trading_consumption"]

        self._last_weight = np.zeros((self._coin_number+1,))
        self._last_weight[0] = 1.0

    @property
    def agent(self):
        # Getter for the agent attribute
        return self._agent

    @property
    def agent_algorithm(self):
        # Getter for the agent algorithm attribute
        return self._agent_alg

    @property
    def test_pv(self):
        # Getter for the portfolio value attribute
        return self._test_pv

    @property
    def test_data(self):
        # Getter for the test data attribute, which generates the test data
        return self._generate_test_data()

    @property
    def test_pc_vector(self):
        # Getter for the portfolio change vector
        return np.array(self._test_pc_vector, dtype=np.float32)

    def trade(self):
        """
        This method simulates trading. It loops through each time step in the test set,
        makes a decision based on the current history and last weights, performs a trade based on the decided weights,
        and then logs the results of the trade.
        """
        logging.info("Running algorithm: {}".format(self._agent_alg))
        while self._steps < self._test_set_length:
            weight = self._agent.decide_by_history(self._generate_history(),
                                                   self._last_weight.copy(),
                                                   test_data=
                                                   self._generate_test_data())
            portfolio_change, total_capital, last_weight = \
                self._trade_by_strategy(weight)

            self._total_capital = total_capital
            self._last_weight = last_weight
            self._test_pc_vector.append(portfolio_change)

            if self._verbose:
                logging.info("""
                =============================================================
                Step {}:
                Raw weights:       {}
                Total assets:      {:.3f} BTC
                Portfolio change:  {:.5f}
                """.format(
                    self._steps + 1,
                    ",".join(
                        ["{:.2e}:{}".format(w, c)
                         for w, c in zip(weight,
                                         ["BTC"] + self._coin_name_list)]
                    ),
                    total_capital, portfolio_change
                ))
            self._steps += 1
        self._test_pv = self._total_capital

    def _generate_history(self):
        """
        This method generates the history of inputs for the current time step.
        It gets the current inputs from the test set and normalizes them if the agent isn't a neural network.
        """
        inputs = self._test_set_X[self._steps]
        if self._agent_alg != "nn":
            # normalize portfolio features with features from the last period.
            inputs = np.concatenate([np.ones([1, 1, inputs.shape[2]]), inputs],
                                    axis=1)
            inputs = inputs[:, :, 1:] / inputs[:, :, :-1]
        return inputs

    def _generate_test_data(self):
        """
        This method generates the test data for the current time step.
        It gets the price relative vectors from the test set and adds a row of ones at the top for BTC.
        """
        test_set = self._test_set_y[:, 0, :].T
        test_set = np.concatenate((np.ones((1, test_set.shape[1])), test_set),
                                  axis=0)
        return test_set

    def _trade_by_strategy(self, weight):
        """
        This method performs a simulated trade based on the provided weight vector. 
        It calculates the portfolio value after commission, the change in the portfolio, the total capital, 
        and the last weight vector after the trade.
        """
        future_price = np.concatenate([np.ones(1),
                                       self._test_set_y[self._steps, 0, :]])
        pv_after_commission = self._calculate_pv_after_commission(
            weight, self._last_weight, self._commission_rate
        )
        portfolio_change = pv_after_commission * np.dot(weight, future_price)
        total_capital = self._total_capital * portfolio_change
        last_weight = (
                pv_after_commission
                * weight
                * future_price
                / portfolio_change
        )
        return portfolio_change, total_capital, last_weight

    @staticmethod
    def _calculate_pv_after_commission(w1, w0, commission_rate):
        """
        This static method calculates the portfolio value after accounting for commission. 
        It uses an iterative process to account for the transaction costs when rebalancing the portfolio.
        Args:
            w1: target portfolio vector, first element is btc.
            w0: rebalanced last period portfolio vector, first element is btc.
            commission_rate: rate of commission fee, proportional to the
            transaction cost.
        """
        mu0 = 1
        mu1 = 1 - 2*commission_rate + commission_rate ** 2
        while abs(mu1-mu0) > 1e-10:
            mu0 = mu1
            mu1 = (1 - commission_rate * w0[0] -
                (2 * commission_rate - commission_rate ** 2) *
                np.sum(np.maximum(w0[1:] - mu1*w1[1:], 0))) / \
                (1 - commission_rate * w1[0])
        return mu1

    @staticmethod
    def _find_latest_checkpoint(model_dir):
        """
        This static method finds the latest checkpoint in the model directory. 
        It raises an error if the directory doesn't exist or there's no checkpoint in it.
        """
        if not os.path.exists(model_dir) or not os.path.isdir(model_dir):
            raise RuntimeError("Model directory doesn't exist!")
        models = os.listdir(model_dir)
        latest_version = -1
        latest_checkpoint = None
        for m in models:
            match = re.fullmatch("epoch=([0-9]+).*$", m)
            if match is not None:
                if int(match.group(1)) > latest_version:
                    latest_checkpoint = m
        if latest_checkpoint is None:
            raise RuntimeError("Checkpoint not found in target directory: {}"
                               .format(model_dir))
        return latest_checkpoint