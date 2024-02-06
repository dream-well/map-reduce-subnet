# The MIT License (MIT)
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

# Bittensor Miner Template:

# Step 1: Import necessary libraries and modules
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

# import miner

processes = {

}

def get_config():
    # Step 2: Set up the configuration parser
    # This function initializes the necessary command-line arguments.
    # Using command-line arguments allows users to customize various miner settings.
    parser = argparse.ArgumentParser()
    parser.add_argument( '--netuid', type = int, default = 10, help = "The chain subnet uid." )
    parser.add_argument( '--axon.port', type = int, default = 8091, help = "Default port" )
    parser.add_argument ( '--port.range', type = str, default = '9000:9010', help = "Opened Port range" )
    parser.add_argument( '--auto_update', default = 'yes', help = "Auto update" ) # yes, no
    # Adds subtensor specific arguments i.e. --subtensor.chain_endpoint ... --subtensor.network ...
    bt.subtensor.add_args(parser)
    # Adds logging specific arguments i.e. --logging.debug ..., --logging.trace .. or --logging.logging_dir ...
    bt.logging.add_args(parser)
    # Adds wallet specific arguments i.e. --wallet.name ..., --wallet.hotkey ./. or --wallet.path ...
    bt.wallet.add_args(parser)
    # Adds axon specific arguments i.e. --axon.port ...
    bt.axon.add_args(parser)
    # Activating the parser to read any command-line inputs.
    # To print help message, run python3 template/miner.py --help
    config = bt.config(parser)

    # Step 3: Set up logging directory
    # Logging captures events for diagnosis or understanding miner's behavior.
    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            'miner',
        )
    )
    # Ensure the directory for logging exists, else create one.
    if not os.path.exists(config.full_path): os.makedirs(config.full_path, exist_ok=True)
    return config


# Main takes the config and starts the miner.
def main( config ):

    # Activating Bittensor's logging with the set configurations.
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(f"Running miner for subnet: {config.netuid} on network: {config.subtensor.chain_endpoint} with config:")

    # This logs the active configuration to the specified logging directory for review.
    bt.logging.info(config)

    # Step 4: Initialize Bittensor miner objects
    # These classes are vital to interact and function within the Bittensor network.
    bt.logging.info("Setting up bittensor objects.")

    # Wallet holds cryptographic information, ensuring secure transactions and communication.
    wallet = bt.wallet( config = config )
    bt.logging.info(f"Wallet: {wallet}")

    # subtensor manages the blockchain connection, facilitating interaction with the Bittensor blockchain.
    subtensor = bt.subtensor( config = config )
    bt.logging.info(f"Subtensor: {subtensor}")

    # metagraph provides the network's current state, holding state about other participants in a subnet.
    metagraph = subtensor.metagraph(config.netuid)
    # bt.logging.info(f"Metagraph: {metagraph} {metagraph.axons}")

    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        bt.logging.error(f"\nYour miner: {wallet} if not registered to chain connection: {subtensor} \nRun btcli register and try again. ")
        os._exit(0)
    else:
        # Each miner gets a unique identity (UID) in the network for differentiation.
        my_subnet_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
        bt.logging.info(f"Running miner on uid: {my_subnet_uid}")

    # Step 4: Set up miner functionalities
    # The following functions control the miner's response to incoming requests.
    # The blacklist function decides if a request should be ignored.
    def blacklist_fn( synapse: protocol.Join ) -> Tuple[bool, str]:
        # Runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        # The synapse is instead contructed via the headers of the request. It is important to blacklist
        # requests before they are deserialized to avoid wasting resources on requests that will be ignored.
        # Below: Check that the hotkey is a registered entity in the metagraph.
        if synapse.dendrite.hotkey not in metagraph.hotkeys:
            # Ignore requests from unrecognized entities.
            bt.logging.trace(f'Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}')
            return True, ""
        caller_uid = metagraph.hotkeys.index( synapse.dendrite.hotkey ) # Get the caller index.
        stake = float( metagraph.S[ caller_uid ] ) # Return the stake as the priority.
        bt.logging.info(f"Stake: {stake}")
        if stake < 3000:
            bt.logging.trace(f'Blacklisting hotkey {synapse.dendrite.hotkey} without enough stake')
            return True, ""
        return False, ""

    # The priority function determines the order in which requests are handled.
    # More valuable or higher-priority requests are processed before others.
    def priority_fn( synapse: protocol.Join ) -> float:
        caller_uid = metagraph.hotkeys.index( synapse.dendrite.hotkey ) # Get the caller index.
        prirority = float( metagraph.S[ caller_uid ] ) # Return the stake as the priority.
        bt.logging.trace(f'Prioritizing {synapse.dendrite.hotkey} with value: ', prirority)
        return prirority

    # This is the core miner function, which decides the miner's response to a valid, high-priority request.
    def get_miner_status( synapse: protocol.MinerStatus ) -> protocol.MinerStatus:
        # Check version of the synapse
        validator_uid = metagraph.hotkeys.index( synapse.dendrite.hotkey )
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
        synapse.available = not utils.is_process_running(processes)
        return synapse

    # The following functions control the miner's response to incoming requests.
    # The blacklist function decides if a request should be ignored.
    def blacklist_miner_status( synapse: protocol.MinerStatus ) -> Tuple[bool, str]:
        if synapse.dendrite.hotkey not in metagraph.hotkeys:
            # Ignore requests from unrecognized entities.
            bt.logging.trace(f'Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}')
            return True, ""
        caller_uid = metagraph.hotkeys.index( synapse.dendrite.hotkey ) # Get the caller index.
        stake = float( metagraph.S[ caller_uid ] ) # Return the stake as the priority.
        bt.logging.info(f"Stake: {stake}")
        if stake < 3000:
            bt.logging.trace(f'Blacklisting hotkey {synapse.dendrite.hotkey} without enough stake')
            return True, ""
        return False, ""

    def join_group( synapse: protocol.Join ) -> protocol.Join:
        validator_uid = metagraph.hotkeys.index( synapse.dendrite.hotkey )
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
            synapse.job.rank = synapse.ranks.get(str(my_subnet_uid))
            if synapse.job.client_hotkey in processes and processes[synapse.job.client_hotkey]['process'].is_alive():
                synapse.joining = False
                synapse.reason = 'Already in group'
                return synapse
            # try to join the group
            bt.logging.info("🔵 Start Process ...")
            queue = mp.Queue()
            process = mp.Process(target=start_miner_dist_process, args=(queue, axon.external_ip, config.port.range, wallet, synapse.job))
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
            bt.logging.info(f"❌ Error {e}")
            traceback.print_exc()
            synapse.joining = False
            synapse.reason = str(e)
            return synapse


    # The following functions control the miner's response to incoming requests.
    # The blacklist function decides if a request should be ignored.
    def blacklist_speed_test( synapse: protocol.SpeedTest ) -> Tuple[bool, str]:
        # Runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        # The synapse is instead contructed via the headers of the request. It is important to blacklist
        # requests before they are deserialized to avoid wasting resources on requests that will be ignored.
        # Below: Check that the hotkey is a registered entity in the metagraph.
        if synapse.dendrite.hotkey not in metagraph.hotkeys:
            # Ignore requests from unrecognized entities.
            bt.logging.trace(f'Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}')
            return True, ""
        caller_uid = metagraph.hotkeys.index( synapse.dendrite.hotkey ) # Get the caller index.
        stake = float( metagraph.S[ caller_uid ] ) # Return the stake as the priority.
        bt.logging.info(f"Stake: {stake}")
        if stake < 3000:
            bt.logging.trace(f'Blacklisting hotkey {synapse.dendrite.hotkey} without enough stake')
            return True, ""
        return False, ""

    # The priority function determines the order in which requests are handled.
    # More valuable or higher-priority requests are processed before others.
    def priority_speed_test( synapse: protocol.SpeedTest ) -> float:
        caller_uid = metagraph.hotkeys.index( synapse.dendrite.hotkey ) # Get the caller index.
        prirority = float( metagraph.S[ caller_uid ] ) # Return the stake as the priority.
        bt.logging.trace(f'Prioritizing {synapse.dendrite.hotkey} with value: ', prirority)
        return prirority

    # This is the core miner function, which decides the miner's response to a valid, high-priority request.
    def speed_test( synapse: protocol.SpeedTest ) -> protocol.SpeedTest:
        # Check version of the synapse
        validator_uid = metagraph.hotkeys.index( synapse.dendrite.hotkey )
        bt.logging.info(f"Validator {validator_uid} asks speed test")
        if not utils.check_version(synapse.version):
            synapse.version = utils.get_my_version()
            return synapse
        
        # Speed Test
        synapse.result = speedtest()
        if synapse.result:
            bt.logging.info("Download: " + str(round(synapse.result['download']['bandwidth'] * 8 / 1000000, 2)) + " Mbps")
            bt.logging.info("Upload: " + str(round(synapse.result['upload']['bandwidth'] * 8 / 1000000, 2)) + " Mbps")
            bt.logging.info("Ping: " + str(synapse.result['ping']['latency']) + " ms")
            bt.logging.info("URL: " + synapse.result['result']['url'])
        
        synapse.version = utils.get_my_version()
        return synapse

    # Step 5: Build and link miner functions to the axon.
    # The axon handles request processing, allowing validators to send this process requests.
    axon = bt.axon( wallet = wallet, config = config,  port = config.axon.port )
    bt.logging.info(f"Axon {axon}")

    # Attach determiners which functions are called when servicing a request.
    bt.logging.info(f"Attaching forward function to axon.")
    axon.attach(
        forward_fn = get_miner_status,
        blacklist_fn = blacklist_miner_status,
    ).attach(
        forward_fn = speed_test,
        blacklist_fn = blacklist_speed_test,
        priority_fn = priority_speed_test
    ).attach(
        forward_fn = join_group,
        blacklist_fn = blacklist_fn,
        priority_fn = priority_fn
    )

    # Serve passes the axon information to the network + netuid we are hosting on.
    # This will auto-update if the axon port of external ip have changed.
    bt.logging.info(f"Serving axon on network: {config.subtensor.chain_endpoint} with netuid: {config.netuid}")
    axon.serve( netuid = config.netuid, subtensor = subtensor )

    # Start  starts the miner's axon, making it active on the network.
    bt.logging.info(f"Starting axon server on port: {config.axon.port}")
    axon.start()

    thread = mp.Process(target=utils.check_processes, args=(processes,))
    thread.start()
    
    # Step 6: Keep the miner alive
    # This loop maintains the miner's operations until intentionally stopped.
    bt.logging.info(f"Starting main loop")
    step = 0
    while True:
        try:
            # Below: Periodically update our knowledge of the network graph.
            if step % 5 == 0:
                metagraph = subtensor.metagraph(config.netuid)
                log =  (f'Step:{step} | '\
                        f'Block:{metagraph.block.item()} | '\
                        f'Stake:{metagraph.S[my_subnet_uid]} | '\
                        f'Rank:{metagraph.R[my_subnet_uid]} | '\
                        f'Trust:{metagraph.T[my_subnet_uid]} | '\
                        f'Consensus:{metagraph.C[my_subnet_uid] } | '\
                        f'Incentive:{metagraph.I[my_subnet_uid]} | '\
                        f'Emission:{metagraph.E[my_subnet_uid]}')
                bt.logging.info(log)
                
            # Check for auto update
            if step % 5 == 0 and config.auto_update != "no":
                utils.update_repository()
            
            step += 1
            time.sleep(bt.__blocktime__)

        # If someone intentionally stops the miner, it'll safely terminate operations.
        except KeyboardInterrupt:
            axon.stop()
            bt.logging.success('Miner killed by keyboard interrupt.')
            break
        # In case of unforeseen errors, the miner will log the error and continue operations.
        except Exception as e:
            bt.logging.error(traceback.format_exc())
            continue


# This is the main function, which runs the miner.
# ... (rest of the code)

if __name__ == "__main__":
    world_size = 16  # Total number of processes
    base_port = 8091  # Starting port for the first miner instance
    processes = []

    for rank in range(world_size):
        # Calculate a unique port for this miner instance
        unique_port = base_port + rank

        # Create a new process targeting the start_distributed_miner function
        p = mp.Process(target=start_miner_dist_process, args=(rank, worker_url, unique_port, wallet, job, axon_port))


        p.start()
        processes.append(p)

    for p in processes:
        p.join()  # Wait for all processes to complete

