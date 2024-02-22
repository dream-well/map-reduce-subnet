# The MIT License (MIT)
# Copyright © 2023 ChainDude

import os
import time
import argparse
import traceback
import bittensor as bt
from typing import Tuple
import torch.multiprocessing as mp
from dist_miner import start_miner_dist_process
from mapreduce import utils, protocol
from speedtest import speedtest
from threading import Thread

processes = {

}

def get_config():

    parser = argparse.ArgumentParser()
    parser.add_argument( '--netuid', type = int, default = 10, help = "The chain subnet uid." )
    parser.add_argument( '--axon.port', type = int, default = 8091, help = "Default port" )
    parser.add_argument ( '--port.range', type = str, default = '9000:9010', help = "Opened Port range" )
    parser.add_argument( '--auto_update', default = 'on', help = "Auto update" ) # on, off
    parser.add_argument( '--blacklist', default = 'on', help = "Blacklist low stake validator" ) # on, off

    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    bt.axon.add_args(parser)
    config = bt.config(parser)

    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            'miner',
        )
    )
    if not os.path.exists(config.full_path): os.makedirs(config.full_path, exist_ok=True)
    return config


    
    # Step 6: Keep the miner alive
    
class Miner:
    
    config: bt.config
    subtensor: bt.subtensor
    axon: bt.axon
    metagraph: bt.metagraph
    wallet: bt.wallet
    my_uid: int
    
    is_speedtest_running = False
    last_speedtest = 0
    
    def __init__(self, config):
        self.config = config

        # Activating Bittensor's logging with the set configurations.
        bt.logging(config=config, logging_dir=config.full_path)
        bt.logging.info(f"Running miner for subnet: {config.netuid} on network: {config.subtensor.chain_endpoint} with config:")

        # This logs the active configuration to the specified logging directory for review.
        bt.logging.info(config)

        self.wallet = bt.wallet( config = config )
        self.subtensor = bt.subtensor( config = config )
        self.metagraph = self.subtensor.metagraph(config.netuid)
        self.axon = bt.axon( wallet = self.wallet, config = config,  port = config.axon.port )

        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            bt.logging.error(f"\nYour miner: {self.wallet} if not registered to chain connection: {self.subtensor} \nRun btcli register and try again. ")
            os._exit(0)
        else:
            # Each miner gets a unique identity (UID) in the network for differentiation.
            self.my_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
            bt.logging.info(f"Running miner on uid: {self.my_uid}")

    def sync(self):
        self.metagraph = self.subtensor.metagraph(config.netuid)

    def blacklist_miner_status( self, synapse: protocol.MinerStatus ) -> Tuple[bool, str]:
        allowed_hotkeys = [
            "5DkRd1V7eurDpgKbiJt7YeJzQvxgpPiiU6FMDf8RmYQ78DpD", # Allow subnet owner's hotkey to fetch Miner Status
        ]
        if synapse.dendrite.hotkey in allowed_hotkeys:
            return False, ""
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            # Ignore requests from unrecognized entities.
            bt.logging.trace(f'Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}')
            return True, ""
        caller_uid = self.metagraph.hotkeys.index( synapse.dendrite.hotkey ) # Get the caller index.
        stake = float( self.metagraph.S[ caller_uid ] ) # Return the stake as the priority.
        bt.logging.info(f"Stake: {stake}")
        if stake < 3000 and self.config.blacklist == "on":
            bt.logging.trace(f'Blacklisting hotkey {synapse.dendrite.hotkey} without enough stake')
            return True, ""
        return False, ""

    # This is the core miner function, which decides the miner's response to a valid, high-priority request.
    def get_miner_status( self, synapse: protocol.MinerStatus ) -> protocol.MinerStatus:
        # Check version of the synapse
        validator_uid = self.metagraph.hotkeys.index( synapse.dendrite.hotkey )
        bt.logging.info(f"Validator {validator_uid} asks Miner Status")
        if not utils.check_version(synapse.version):
            synapse.version = utils.get_my_version()
            return synapse
        # Get Free Memory and Calculate Bandwidth
        synapse.free_memory = utils.get_available_memory()
        bt.logging.info(f"Free memory: {utils.human_readable_size(synapse.free_memory)}")
        synapse.version = utils.get_my_version()
        
        if utils.update_flag:
            synapse.available = False
            return synapse
        synapse.available = True
        # not utils.is_process_running(processes)
        return synapse

    def blacklist_join( self, synapse: protocol.Join ) -> Tuple[bool, str]:
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            bt.logging.trace(f'Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}')
            return True, ""
        caller_uid = self.metagraph.hotkeys.index( synapse.dendrite.hotkey ) # Get the caller index.
        stake = float( self.metagraph.S[ caller_uid ] ) # Return the stake as the priority.
        bt.logging.info(f"Stake: {stake}")
        if stake < 3000 and self.config.blacklist == "on":
            bt.logging.trace(f'Blacklisting hotkey {synapse.dendrite.hotkey} without enough stake')
            return True, ""
        return False, ""

    def join_group( self, synapse: protocol.Join ) -> protocol.Join:
        validator_uid = self.metagraph.hotkeys.index( synapse.dendrite.hotkey )
        bt.logging.info(f"Validator {validator_uid} asks Joining Group")
        try:
            if not utils.check_version(synapse.version):
                synapse.version = utils.get_my_version()
                return synapse
            if utils.update_flag:
                synapse.joining = False
                synapse.reason = 'Update'
                return synapse
            if utils.is_process_running(processes):
                synapse.joining = False
                synapse.reason = 'Working'
                return synapse
            synapse.version = utils.get_my_version()
            synapse.job.rank = synapse.ranks.get(str(self.my_uid))
            if synapse.job.client_hotkey in processes and processes[synapse.job.client_hotkey]['process'].is_alive():
                synapse.joining = False
                synapse.reason = 'Already in group'
                return synapse
            # try to join the group
            bt.logging.info("🔵 Start Process ...")
            queue = mp.Queue()
            process = mp.Process(target=start_miner_dist_process, args=(queue, self.axon.external_ip, config.port.range, self.wallet, synapse.job))
            process.start()
            processes[synapse.job.client_hotkey] = {
                'process': process,
                'queue': queue,
                'job': synapse.job
            }
            synapse.joining = True
            return synapse
        except Exception as e:
            # if failed, set joining to false
            bt.logging.info(f"🛑 Error {e}")
            traceback.print_exc()
            synapse.joining = False
            synapse.reason = str(e)
            return synapse

    # The following functions control the miner's response to incoming requests.
    # The blacklist function decides if a request should be ignored.
    def blacklist_speed_test( self, synapse: protocol.SpeedTest ) -> Tuple[bool, str]:
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            bt.logging.trace(f'Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}')
            return True, ""
        caller_uid = self.metagraph.hotkeys.index( synapse.dendrite.hotkey ) # Get the caller index.
        stake = float( self.metagraph.S[ caller_uid ] ) # Return the stake as the priority.
        bt.logging.info(f"Stake: {stake}")
        if stake < 3000 and self.config.blacklist == "on":
            bt.logging.trace(f'Blacklisting hotkey {synapse.dendrite.hotkey} without enough stake')
            return True, ""
        if not self.speedtest_available():
            bt.logging.trace(f'Blacklisting hotkey {synapse.dendrite.hotkey} while speed testing')
            return True, ""
        return False, ""

    def speedtest_available(self):
        if self.is_speedtest_running or time.time() - self.last_speedtest < 60:
            return False
        return True

    # This is the core miner function, which decides the miner's response to a valid, high-priority request.
    def speed_test( self, synapse: protocol.SpeedTest ) -> protocol.SpeedTest:
        # Check version of the synapse
        validator_uid = self.metagraph.hotkeys.index( synapse.dendrite.hotkey )
        bt.logging.info(f"Validator {validator_uid} asks speed test")
        if not utils.check_version(synapse.version):
            synapse.version = utils.get_my_version()
            return synapse
        
        # Speed Test
        self.is_speedtest_running = True
        try:
            bt.logging.info("🔵 Start Speed Test ...")
            synapse.result = speedtest()
        except Exception as e:
            bt.logging.error(f"Error: {e}")
            synapse.result = None
        self.is_speedtest_running = False
        self.last_speedtest = time.time()
        if synapse.result:
            bt.logging.info("Download: " + str(round(synapse.result['download']['bandwidth'] * 8 / 1000000, 2)) + " Mbps")
            bt.logging.info("Upload: " + str(round(synapse.result['upload']['bandwidth'] * 8 / 1000000, 2)) + " Mbps")
            bt.logging.info("Ping: " + str(synapse.result['ping']['latency']) + " ms")
            bt.logging.info("URL: " + synapse.result['result']['url'])
        
        synapse.version = utils.get_my_version()
        return synapse
    
    def start_axon(self):
        # Attach determiners which functions are called when servicing a request.
        bt.logging.info(f"Attaching forward function to axon.")
        self.axon.attach(
            forward_fn = self.get_miner_status,
            blacklist_fn = self.blacklist_miner_status,
        ).attach(
            forward_fn = self.speed_test,
            blacklist_fn = self.blacklist_speed_test,
        ).attach(
            forward_fn = self.join_group,
            blacklist_fn = self.blacklist_join,
        )
        
        bt.logging.info(f"Serving axon on network: {self.config.subtensor.chain_endpoint} with netuid: {config.netuid}")
        self.axon.serve( netuid = self.config.netuid, subtensor = self.subtensor )
        
        bt.logging.info(f"Starting axon server on port: {config.axon.port}")
        self.axon.start()

    def run(self):
        # start axon
        self.start_axon()
        
        self.thread = Thread(target=utils.check_processes, args=(processes,))
        self.thread.start()
        
        # This loop maintains the miner's operations until intentionally stopped.
        bt.logging.info(f"Starting main loop")
        step = 0
        while True:
            try:
                # Below: Periodically update our knowledge of the network graph.
                if step % 5 == 0:
                    self.sync()
                    bt.logging.info(f'Step:{step} | Block:{self.metagraph.block.item()} | Trust:{self.metagraph.T[self.my_uid]} | Incentive:{self.metagraph.I[self.my_uid]} | Emission:{self.metagraph.E[self.my_uid]}')
                    
                # Check for auto update
                if step % 5 == 0 and config.auto_update != "no":
                    utils.update_repository()
                
                step += 1
                time.sleep(bt.__blocktime__)

            # If someone intentionally stops the miner, it'll safely terminate operations.
            except KeyboardInterrupt:
                self.axon.stop()
                self.thread.join(2)
                bt.logging.success('Miner killed by keyboard interrupt.')
                break
            
            # In case of unforeseen errors, the miner will log the error and continue operations.
            except Exception as e:
                bt.logging.error(traceback.format_exc())
                continue


# This is the main function, which runs the miner.
if __name__ == "__main__":
    config = get_config()
    miner = Miner(config)
    miner.run()
