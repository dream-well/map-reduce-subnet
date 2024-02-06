# Import necessary libraries and modules
import os
import argparse
import bittensor as bt
import torch.multiprocessing as mp
import torch
import json
import time

def get_validator_config_from_json():
    with open('validator.config.json') as f:
        validator_config = json.load(f)
    return validator_config

# Load the validator configuration from the JSON file
validator_config = get_validator_config_from_json()

def get_config():
    """
    Sets up the configuration parser and initializes necessary command-line arguments.
    
    Returns:
        config (Namespace): Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    # Adds override arguments for network and netuid.
    parser.add_argument( '--subtensor.network', default = validator_config.get('subtensor.network', 'finney'), help = "The subtensor network." )
    parser.add_argument( '--netuid', type = int, default = validator_config.get('netuid', 10), help = "The chain subnet uid." )
    parser.add_argument( '--wallet.name', default = validator_config.get('wallet.name', 'default'), help = "Wallet name" )
    parser.add_argument( '--wallet.hotkey', default = validator_config.get('wallet.hotkey', 'default'), help = "Wallet hotkey" )
    parser.add_argument( '--auto_update', default = validator_config.get('auto_update', 'yes'), help = "Auto update" ) # yes, no
    # Adds subtensor specific arguments i.e. --subtensor.chain_endpoint ... --subtensor.network ...
    bt.subtensor.add_args(parser)
    # Adds logging specific arguments i.e. --logging.debug ..., --logging.trace .. or --logging.logging_dir ...
    bt.logging.add_args(parser)
    # Adds wallet specific arguments i.e. --wallet.name ..., --wallet.hotkey ./. or --wallet.path ...
    bt.wallet.add_args(parser)
    # Adds axon specific arguments i.e. --axon.port ...
    bt.axon.add_args(parser)

    # Activating the parser to read any command-line inputs.
    config = bt.config(parser)

    # Step 3: Set up logging directory
    # Logging captures events for diagnosis or understanding validator's behavior.
    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            'validator',
        )
    )
    # Ensure the directory for logging exists, else create one.
    if not os.path.exists(config.full_path): os.makedirs(config.full_path, exist_ok=True)
    return config

# Global variable to last benchmark time
last_benchmark_at = 0

# Global variable to store validator status
status = {}
    
# Main takes the config and starts the validator.
def main( config ):

    # Activating Bittensor's logging with the set configurations.
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(f"Running validator for subnet: {config.netuid} on network: {config.subtensor.chain_endpoint} with config:")

    # This logs the active configuration to the specified logging directory for review.
    bt.logging.info(config)

    # Step 4: Initialize Bittensor validator objects
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
    bt.logging.info(f"Metagraph: {metagraph} {metagraph.axons}")
    
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        bt.logging.error(f"\nYour validator: {wallet} if not registered to chain connection: {subtensor} \nRun btcli register and try again. ")
        time.sleep(3)
        os._exit(0)
    else:
        # Each validator gets a unique identity (UID) in the network for differentiation.
        my_subnet_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
        bt.logging.info(f"Running validator on uid: {my_subnet_uid}")

    # Step 5: Build and link validator functions to the axon.
    # The axon handles request processing, allowing validators to send this process requests.
    axon = bt.axon( wallet = wallet, config = config, port = config.axon.port )
    bt.logging.info(f"Axon {axon}")

    # Serve passes the axon information to the network + netuid we are hosting on.
    # This will auto-update if the axon port of external ip have changed.
    bt.logging.info(f"Serving axon on network: {config.subtensor.chain_endpoint} with netuid: {config.netuid}")

    # Step 6: Keep the validator alive
    # This loop maintains the validator's operations until intentionally stopped.
    bt.logging.info(f"Starting main loop")

    scores = torch.ones_like(metagraph.S, dtype=torch.float32)
    bt.logging.info(f"Weights: {scores}")
    
    scores_file = "scores.pt"
    try:
        scores = torch.load(scores_file)
        bt.logging.info(f"Loaded scores from save file: {scores}")
    except:
        scores = torch.zeros_like(metagraph.S, dtype=torch.float32)
        bt.logging.info(f"Initialized all scores to 0")

    weights = torch.nn.functional.normalize(scores, p=1.0, dim=0)
    bt.logging.info(f"Setting weights: {weights}")
    
    while True:
        
        # This is a crucial step that updates the incentive mechanism on the Bittensor blockchain.
        # Miners with higher scores (or weights) receive a larger share of TAO rewards on this subnet.
        result = subtensor.set_weights(
            netuid = config.netuid, # Subnet to set weights on.
            wallet = wallet, # Wallet to sign set weights using hotkey.
            uids = metagraph.uids, # Uids of the miners to set weights for.
            weights = weights, # Weights to set for the miners. 
            wait_for_inclusion=True,
            ttl = 60
        )
        
        if result: 
            bt.logging.info(f'result: {result}')
            bt.logging.success('✅ Successfully set weights.')
            break
            
        else: bt.logging.error('Failed to set weights.')    
                

# This is the main function, which runs the miner.
if __name__ == "__main__":
    mp.set_start_method('spawn')  # This is often necessary in PyTorch multiprocessing
    main( get_config() )
