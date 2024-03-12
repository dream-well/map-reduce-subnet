# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 ChainDude

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import time

# Bittensor
import bittensor as bt

# Bittensor Validator:
import taomap
from taomap.validator import forward

# import base validator class which takes care of most of the boilerplate
from taomap.base.validator import BaseValidatorNeuron
import datetime as dt
import wandb
import constants
import random
from sklearn.cluster import KMeans
import numpy as np
import os
import json
import functools
import taomap.utils as utils
import traceback
import threading

class Validator(BaseValidatorNeuron):
    """
    Your validator neuron class. You should use this class to define your validator's behavior. In particular, you should replace the forward function with your own logic.

    This class inherits from the BaseValidatorNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a validator such as keeping a moving average of the scores of the miners and using them to set weights at the end of each epoch. Additionally, the scores are reset for new hotkeys at the end of each epoch.
    """

    def __init__(self, config=None):

        self.term = 0
        self.init_term_variables()
        
        super(Validator, self).__init__(config=config)

        bt.logging.info(self.config)
        # Dont log to wandb if offline.
        # if not self.config.offline and self.config.wandb.on:
        self.new_wandb_run()
        self.subtensor_benchmark = bt.subtensor(config=self.config)

    def init_term_variables(self):
        self.is_seedhash_commited = False
        self.is_seed_commited = False
        self.is_set_weight = False
        self.is_seed_shared = False
        if not hasattr(self, 'next_seed'):
            self.next_seed = 0
        self.seed = self.next_seed
        self.next_seed = random.randint(0, 100)
        self.voted_uid = None
        self.voted_groups = []
        self.groups = None
        self.is_uploaded_group = False
        self.benchmark_state = {}
        self.benchmark_thread = None

    def update_term_bias(self):
        self.block_height = self.subtensor.get_current_block()
        self.current_term = (self.block_height - constants.ORIGIN_TERM_BLOCK) // constants.BLOCKS_PER_TERM
        self.term_bias = (self.block_height - constants.ORIGIN_TERM_BLOCK) % constants.BLOCKS_PER_TERM

    async def forward(self):
        axon = self.metagraph.axons[3]
        synapse = taomap.protocol.Benchmark(shape=list(constants.BENCHMARK_SHAPE))
        benchmark_at = time.time()
        bt.logging.info("Sending benchmark request to miner 3")
        responses = self.dendrite.query([axon], synapse, timeout = 120, deserialize = True)
        if responses is None:
            bt.logging.info(f"Response from {3}: None")
            return
        [arrived_at, size] = responses[0]
        bt.logging.info(f"Response from {3}: {(size / 1024 / 1024):.2f} MB, time: {arrived_at - benchmark_at}")
        return

    async def forward1(self):
        """
        Validator forward pass. Consists of:
        - Generating the query
        - Querying the miners
        - Getting the responses
        - Rewarding the miners
        - Updating the scores
        """
        try: 
            self.update_term_bias()
            bt.logging.info(f"Current block height: {self.block_height}, current term: {self.current_term}, blocks: {self.term_bias}")
            if self.current_term > self.term:
                bt.logging.info(f"New term {self.current_term}")
                self.term = self.current_term
                self.init_term_variables()

            # Commit hash of the next term seed
            if self.term_bias >= constants.BLOCKS_SEEDHASH_START and self.term_bias < constants.BLOCKS_SEEDHASH_END:
                if not self.is_seedhash_commited:
                    self.is_seedhash_commited = self.commit_data({
                        "type": "seedhash",
                        "term": self.term + 1,
                        "seedhash": hash(str(self.next_seed))
                    })
                    bt.logging.info(f"Committed seed hash for term {self.term + 1}")
                    self.update_term_bias()
            
            # Commit seed of the current term
            if self.term_bias < constants.BLOCKS_SHARE_SEED:
                # If groups are not uploaded, upload them
                if self.groups == None:
                    self.groups = self.cluster_miners()
                if not self.is_uploaded_group:
                    self.is_uploaded_group = self.upload_state()
                # Commit seed
                if self.is_uploaded_group and not self.is_seed_commited:
                    self.is_seed_commited = self.commit_data({
                            "type": "seed",
                            "term": self.term,
                            "seedhash": hash(str(self.seed)),
                            "seed": self.seed,
                            "grouphash": hash(str(self.groups))
                        })
                    bt.logging.info(f"Committed seed for term {self.term}")
                self.update_term_bias()
                                
            # Get all validator's commits and groups, seeds.
            if self.term_bias >= constants.BLOCKS_SHARE_SEED and self.term_bias <= constants.BLOCKS_START_BENCHMARK:
                if self.voted_uid is None:
                    self.voted_uid, self.voted_groups = self.get_vote_result()
            
            # Benchmark
            if self.term_bias >= constants.BLOCKS_START_BENCHMARK and self.term_bias < constants.BLOCKS_SEEDHASH_START:
                if self.benchmark_thread is None or not self.benchmark_thread.is_alive():
                    self.start_benchmark_thread()
                return

            return await forward(self)
        except BaseException as e:
            bt.logging.error(f"Error committing: {e}")
            bt.logging.debug(traceback.format_exc())

    def start_benchmark_thread(self):
        self.benchmark_thread = threading.Thread(target=self.benchmark, daemon=True)
        self.benchmark_thread.start()            

    def benchmark(self):
        bt.logging.info("Benchmarking thread started")
        benchmark_started = False
        while True:
            try:
                current_block = self.subtensor_benchmark.get_current_block()
                term_bias = (current_block - constants.ORIGIN_TERM_BLOCK) % constants.BLOCKS_PER_TERM
                if term_bias < constants.BLOCKS_START_BENCHMARK:
                    time.sleep(1)
                    continue
                if not benchmark_started:
                    benchmark_started = True
                    bt.logging.info("🚀 Benchmarking started")
                current_group_id = (term_bias - constants.BLOCKS_START_BENCHMARK) // constants.BLOCKS_PER_GROUP
                if self.voted_uid is None:
                    bt.logging.warning("No voted uid")
                    time.sleep(2)
                    break
                if current_group_id >= len(self.voted_groups):
                    bt.logging.info(f"No group info for {current_group_id}")
                    time.sleep(2)
                    continue
                current_group = self.voted_groups[current_group_id]
                bt.logging.info(f"Benchmarking group {current_group_id}: {current_group}")

                axons = [self.metagraph.axons[uid] for uid in current_group]
                synapse = taomap.protocol.Benchmark(shape=list(constants.BENCHMARK_SHAPE))
                benchmark_at = time.time()
                responses = self.dendrite.query(axons, synapse, timeout = 120, deserialize = True)
                for i, uid in enumerate(current_group):
                    response = responses[i]
                    if response is None or response[1] is None:
                        self.benchmark_state[uid] = -1
                        continue
                    data: bt.Tensor = response[1]
                    bt.logging.info(f"Response from {uid}: {data.shape}")
                    if data.shape != list(constants.BENCHMARK_SHAPE):
                        self.benchmark_state[uid] = -1
                        continue
                    self.benchmark_state[uid] = response[0] - benchmark_at
                    bt.logging.info(f"Benchmark time for {uid}: {self.benchmark_state[uid]}s")

                if current_group_id >= len(self.voted_groups):
                    bt.logging.info("✅ Benchmarking finished")
                    break
                time.sleep(0.1)
            except BaseException as e:
                bt.logging.error(f"Error benchmarking: {e}")
                bt.logging.debug(traceback.format_exc())
                time.sleep(0.1)
        bt.logging.info("Benchmarking thread finished")
    
    def commit_data_mock(self, data: dict[str, any]):
        self._committed_data = data
        self._committed_data['block'] = self.subtensor.get_current_block()
        return True

    def commit_data(self, data: dict[str, any]):
        commit_str = json.dumps(data)
        try:
            self.subtensor.commit(self.wallet, self.config.netuid, commit_str)
            bt.logging.info(f"Committed: {commit_str}")
            return True
        except BaseException as e:
            bt.logging.error(f"Error committing: {e}")
            bt.logging.debug(traceback.format_exc())
            return False
        
    def get_commit_data_mock(self, uid):
        if hasattr(self, '_committed_data'):
            return self._committed_data
        return None

    def get_commit_data(self, uid):
        try:
            metadata = bt.extrinsics.serving.get_metadata(self.subtensor, self.config.netuid, self.hotkeys[uid] )
            if metadata is None:
                return None
            last_commitment = metadata["info"]["fields"][0]
            hex_data = last_commitment[list(last_commitment.keys())[0]][2:]
            data = json.loads(bytes.fromhex(hex_data).decode())
            data['block'] = metadata['block']
            return data
        except BaseException as e:
            bt.logging.error(f"Error getting commitment: {e}")
            bt.logging.debug(traceback.format_exc())
            return None

    def get_vote_result(self):
        # Download all commits and groups, seeds
        validator_uids = [uid for uid in self.metagraph.uids if self.metagraph.stake[uid] >= constants.VALIDATOR_MIN_STAKE]
        bt.logging.info(f"Voting on validators {validator_uids}")
        # Get all commits
        commits = []
        for uid in validator_uids:
            commit_data = self.get_commit_data(uid)
            bt.logging.debug(f"Commit data {uid}: {commit_data}")
            if commit_data is None:
                continue
            if commit_data['term'] != self.term or commit_data['block'] % constants.BLOCKS_PER_TERM > constants.BLOCKS_SHARE_SEED:
                bt.logging.debug(f"{uid} {commit_data} is not valid for term {self.term}")
                continue
            commits.append({
                "uid": uid,
                "term": commit_data["term"],
                "block": commit_data['block'],
                "seedhash": commit_data["seedhash"],
                "seed": commit_data["seed"],
                "grouphash": commit_data["grouphash"]
            })
        print("Commits: ", commits)

        # Get all shared seeds
        for commit in commits:
            try:
                artifact_url = f"{self.config.wandb.entity}/{self.config.wandb.project_name}/state-{commit['uid']}:latest"
                artifact = wandb.use_artifact(artifact_url)
                artifact_dir = artifact.download(self.config.neuron.full_path)
                shared_file = os.path.join(artifact_dir, f"{commit['term']}.json")
                with open(shared_file, 'r') as f:
                    data = json.load(f)
                if commit["seedhash"] != data["hash"]:
                    commit['valid'] = False
                    bt.logging.warning(f"Seed hash mismatch for {commit['uid']}")
                    continue
                commit['valid'] = True
                commit['groups'] = data['groups']
            except BaseException as e:
                commit['valid'] = False
                bt.logging.error(f"Error getting shared seed: {e}")
                bt.logging.debug(traceback.format_exc())

        print("Commits with groups and seeds: ", commits)
        if len(commits) == 0:
            bt.logging.warning("No valid commits")
            return None, []
        valid_commits = [commit for commit in commits if commit['valid']]
        # Vote for the group
        sum_of_seeds = sum(commit['seed'] for commit in valid_commits)
        voted_commit = valid_commits[sum_of_seeds % len(valid_commits)]

        print(f"Voted uid: {voted_commit['uid']}, seed sum: {sum_of_seeds}")
        print(f"Voted groups: {voted_commit['groups']}")

        return voted_commit['uid'], voted_commit['groups']
        
    def upload_state(self):
        """
        Uploads the seed and groups to wandb
        """
        try:
            artifact = wandb.Artifact(f'state-{self.uid}', type = 'dataset')
            file_path = self.config.neuron.full_path + f'/{self.term}.json'
            with open(file_path, 'w') as f:
                json_str = json.dumps({
                    "term": self.term,
                    "seed": self.seed,
                    "hash": hash(str(self.seed)),
                    "groups": self.groups,
                    "grouphash": hash(str(self.groups))
                }, indent=4)
                f.write(json_str)
            artifact.add_file(file_path)
            self.wandb_run.log_artifact(artifact)
            artifact.wait()
            bt.logging.info(f'Uploaded {self.term}.json to wandb')
            return True
        except Exception as e:
            bt.logging.error(f'Error saving seed info: {e}')
            bt.logging.debug(traceback.format_exc())
            return False

    def new_wandb_run(self):
        """Creates a new wandb run to save information to."""
        # Create a unique run id for this run.
        run_id = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name = "validator-" + str(self.uid) + "-" + run_id
        self.wandb_run = wandb.init(
            dir=self.config.neuron.full_path,
            name=name,
            project=self.config.wandb.project_name,
            entity=self.config.wandb.entity,
            config={
                "uid": self.uid,
                "hotkey": self.wallet.hotkey.ss58_address,
                "run_name": run_id,
                "version": constants.__version__,
                "type": "validator",
            },
            allow_val_change=True,
        )

        bt.logging.debug(f"Started a new wandb run: {name}")

    def cluster_miners(self):
        """
        This function is called by the validator every time step.

        Validator should make a benchmark order to the network.

        Miners which have similar ips will be grouped together.

        Each group has 4 miners. Maximum 64 groups are allowed.

        It is responsible for querying the network and scoring the responses.

        Args:
            self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

        """
        def ip_to_int(ip):
            octets = [int(x) for x in ip.split('.')]
            return sum([octets[i] << (24 - 8 * i) for i in range(4)])

        miner_uids = [uid for uid in self.metagraph.uids if self.metagraph.stake[uid] < constants.VALIDATOR_MIN_STAKE and self.metagraph.axons[uid].ip != "0.0.0.0"]

        ips = [self.metagraph.axons[uid].ip for uid in miner_uids]

        # Filter out any duplicate IPs
        unique_ips = set()
        filtered_miner_uids = []

        for uid in self.metagraph.uids:
            ip = self.metagraph.axons[uid].ip
            stake = self.metagraph.stake[uid]
            
            if stake < constants.VALIDATOR_MIN_STAKE and ip != "0.0.0.0" and ip not in unique_ips:
                unique_ips.add(ip)
                filtered_miner_uids.append(uid)

        # Now, filtered_miner_uids contains uids with unique IPs and ips will have those unique IPs
        ips = [self.metagraph.axons[uid].ip for uid in filtered_miner_uids]
        miner_uids = filtered_miner_uids

        numerical_ips = np.array([ip_to_int(ip) for ip in ips]).reshape(-1, 1)
        
        group_count = len(miner_uids) // 4
        # Use K-Means to cluster IPs into 64 groups
        kmeans = KMeans(n_clusters=group_count, random_state=random.randint(0, 100)).fit(numerical_ips)
        labels = kmeans.labels_

        # Group IPs based on labels
        groups = {}
        for i, label in enumerate(labels):
            if label not in groups:
                groups[label] = []
            groups[label].append(ips[i])

        # Convert groups to a list of lists (each sub-list is a group of IPs)
        groups_list = list(groups.values())

        final_groups = []
        leftovers = []

        for group in groups_list:
            if len(group) > 4:
                # Split large groups into groups of 4
                for i in range(0, len(group), 4):
                    new_group = group[i:i+4]
                    if len(new_group) == 4:
                        final_groups.append(new_group)
                    else:
                        leftovers.extend(new_group)
            elif len(group) < 4:
                leftovers.extend(group)
            else:
                final_groups.append(group)

        # Step 2: Merge leftovers to form new groups of 4, if possible
        while len(leftovers) >= 4:
            final_groups.append(leftovers[:4])
            leftovers = leftovers[4:]

        # Display the groups
        for i, group in enumerate(final_groups):
            print(f"Group {i+1}: {group}")
        uid_groups = []
        for group in final_groups:
            uid_group = []
            for ip in group:
                uid_group.append(int(miner_uids[ips.index(ip)]))
            uid_groups.append(uid_group)
        random.shuffle(uid_groups)
        return uid_groups

# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info("Validator running...", time.time())
            time.sleep(5)