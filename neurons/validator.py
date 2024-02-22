# Import necessary libraries and modules
import os
import time
import argparse
import traceback
import bittensor as bt
import torch.multiprocessing as mp
import torch
from typing import Tuple
from mapreduce import utils, protocol
from dist_validator import start_master_process
from threading import Thread
import json
from speedtest import verify_speedtest_result
from datetime import datetime, timezone
from rich.table import Table
from rich.console import Console


def get_validator_config_from_json():
    """
    Reads the validator configuration from a JSON file.
    
    Returns:
        validator_config (dict): Configuration parameters read from the JSON file.
    """
    with open('validator.config.json') as f:
        validator_config = json.load(f)
    return validator_config

# Load the validator configuration from the JSON file
validator_config = get_validator_config_from_json()
is_speedtest_running = False
last_speedtest = 0

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
    parser.add_argument( '--axon.port', type = int, default = 8091, help = "Default port" )
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


# Global variable to store process information
processes = {}

# Global variable to store miner status
miner_status = []

# Global variable to store speed
speedtest_results = {}

# Global variable to last benchmark time
last_benchmark_at = 0
    
# Main takes the config and starts the validator.
def main( config ):

    global status
    global miner_status
    global speedtest_results

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
    
    # Dendrite is the RPC client; it lets us send messages to other nodes (axons) in the network.
    dendrite = bt.dendrite( wallet = wallet )

    # metagraph provides the network's current state, holding state about other participants in a subnet.
    metagraph = subtensor.metagraph(config.netuid)
    
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        bt.logging.error(f"\nYour validator: {wallet} if not registered to chain connection: {subtensor} \nRun btcli register and try again. ")
        os._exit(0)
    else:
        # Each validator gets a unique identity (UID) in the network for differentiation.
        my_subnet_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
        bt.logging.info(f"Running validator on uid: {my_subnet_uid}")

    # Step 5: Build and link validator functions to the axon.
    # The axon handles request processing, allowing validators to send this process requests.
    axon = bt.axon( wallet = wallet, config = config, port = config.axon.port )
    bt.logging.info(f"Axon {axon}")

    def calculate_score():
        scores = torch.zeros_like(metagraph.S, dtype=torch.float32)
        speedtest_scores = torch.zeros_like(metagraph.S, dtype=torch.float32)
        benchmark_scores = torch.zeros_like(metagraph.S, dtype=torch.float32)
        bandwidth_scores = torch.zeros_like(metagraph.S, dtype=torch.float32)
        ip_count = {}
        for miner in miner_status:
            if miner['status'] != "failed":
                uid = miner['uid']
                speedtest_scores[uid] = miner['upload'] * 0.5 + miner['download'] * 0.5
                benchmark_scores[uid] = miner['speed']
                bandwidth_scores[uid] = min(miner['free_memory'], 512 * 1024 * 1024 * 1024 )
                ip = metagraph.neurons[uid].axon_info.ip
                ip_count[ip] = ip_count.get(ip, 0) + 1
        
        # Divide by ip count
        for miner in miner_status:
            if miner['status'] == 'benchmarked':
                ip = metagraph.neurons[uid].axon_info.ip
                speedtest_scores[uid] = speedtest_scores[uid] / ip_count[ip]
                benchmark_scores[uid] = benchmark_scores[uid] / ip_count[ip]
                bandwidth_scores[uid] = bandwidth_scores[uid] / ip_count[ip]
        
        speedtest_scores = torch.nn.functional.normalize(speedtest_scores, p=1.0, dim=0)
        benchmark_scores = torch.nn.functional.normalize(benchmark_scores, p=1.0, dim=0)
        # set bandwidth score to 0 if speed score is 0
        bandwidth_scores = bandwidth_scores * torch.Tensor([benchmark_scores[uid] > 0 for uid in metagraph.uids])
        bandwidth_scores = torch.nn.functional.normalize(bandwidth_scores, p=1.0, dim=0)
        scores = speedtest_scores * 0.55 + benchmark_scores * 0.05 + bandwidth_scores * 0.4
        return scores
    
    def init_miner_status():
        global miner_status
        bt.logging.info(f"Initializing miner status")
        if len(miner_status) == 0:
            for uid, hotkey in enumerate(metagraph.hotkeys):
                miner_status.append({
                    'uid': uid,
                    'hotkey': hotkey,
                    'free_memory': 0,
                    'bandwidth': 0,
                    'speed': 0,
                    'status': 'unavailable',
                    'timestamp': 0,
                    'retry': 0,
                    'upload': 0,
                    'download': 0,
                })
        else:
            for uid, hotkey in enumerate(metagraph.hotkeys):
                if miner_status[uid]['status'] in ['available', 'benchmarked']:
                    miner_status[uid]['status'] = 'available'
                else:
                    miner_status[uid]['status'] = 'unavailable'
                    miner_status[uid]['speed'] = 0
                    miner_status[uid]['bandwidth'] = 0
                    miner_status[uid]['bandwidth_updated_at'] = 0
                miner_status[uid]['retry'] = 0
        save_miner_status()
        
    # Choose miner to benchmark
    def choose_miner():
        for miner in miner_status:
            if miner['status'] == 'available' and miner['retry'] >= 3:
                miner['status'] = 'failed'
        available_miners = [miner for miner in miner_status if miner['status'] == 'available']
        if len(available_miners) == 0:
            return None
        # choose an available miner randomly
        return available_miners[ int(torch.randint(0, len(available_miners), (1,)))]
        
    def update_miner_status():
        
        # try:
        #     here = os.path.dirname(os.path.abspath(__file__))
        #     file_name = os.path.join(here, '../mapreduce/dist/performance')
        #     # Read the exe file and save it to app_data.
        #     with open(file_name, 'rb') as file:
        #         # Read the entire content of the EXE file
        #         app_data = file.read()
        # except Exception as e:
        #     bt.logging.error(f"{e}")
        #     return
        
        # Query the miners for benchmarking
        bt.logging.info(f"🔵 Querying Miner Status")
        responses = dendrite.query(metagraph.axons, protocol.MinerStatus(
            version=utils.get_my_version(),
            # perf_input = repr(app_data),
        ), timeout = 10)
        for response, miner in zip(responses, miner_status):
            if response.available and ( miner['status'] == 'unavailable' or miner['status'] == 'available'):
                # binary_data = ast.literal_eval(response) # Convert str to binary data
                # decoded_data = ast.literal_eval(cipher_suite.decrypt(binary_data).decode()) #Decrypt data and convert it to object
                miner['status'] = 'available'
                miner['timestamp'] = time.time()
                miner['free_memory'] = response.free_memory
        bt.logging.info(f"available miners to benchmark: {[miner['uid'] for miner in miner_status if miner['status'] == 'available']}")
    
    def clear_miner_process(miner):
        if miner.get('process', None) is None:
            return
        del miner['process']
        del miner['output']
        del miner['input']
    
    def check_benchmark_result():
        while True:
            benchmarking_uids = [miner['uid'] for miner in miner_status if miner['status'] == 'benchmarking']
            for miner_uid in benchmarking_uids:
                miner = miner_status[miner_uid]
                if time.time() - miner['timestamp'] > 32:
                    bt.logging.warning(f'🛑 Timeout for benchmark result {miner_uid}')
                    clear_miner_process(miner)
                    miner['status'] = 'available'
                if miner.get('process', None) is None:
                    continue
                if not miner['process'].is_alive():
                    bt.logging.warning(f"🛑 Benchmark process off {miner_uid}")
                    miner['status'] = 'available'
                    clear_miner_process(miner)
                    continue
                if miner['output'].empty():
                    continue
                result:protocol.BenchmarkResult = miner['output'].get()
                log =  (f'🟢 Benchmark Result | UID: {miner_uid} | '\
                        f'Duration: {result.duration:.2f}s | '\
                        f'Data: {utils.human_readable_size(result.data_length)} | '\
                        f'Bandwidth: {utils.human_readable_size(result.bandwidth)} | '\
                        f'Speed: {utils.human_readable_size(result.speed)}/s | '\
                )
                bt.logging.success(log)
                bt.logging.info(f"Benchmarked miners: { len([miner for miner in miner_status if miner['status'] == 'benchmarked'])}")
                miner['input'].put('exit')
                clear_miner_process(miner)
                miner['status'] = 'benchmarked'
                miner['speed'] = result.speed
                miner['timestamp'] = time.time()
                miner['bandwidth'] = result.bandwidth
                miner['bandwidth_updated_at'] = time.time()
                save_miner_status()
                
            time.sleep(0.001)
    
    """
    Process benchmark request
    """
    def request_benchmark( synapse: protocol.RequestBenchmark ) -> protocol.RequestBenchmark:
        
        global last_benchmark_at
        
        last_benchmark_at = time.time()
        
        hotkey = synapse.dendrite.hotkey
        bt.logging.info(f"Benchmark request from {hotkey}")
                
        # Version checking
        if not utils.check_version(synapse.version):
            synapse.version = utils.get_my_version()
            return synapse
        
        if utils.update_flag:
            synapse.job.status = 'error'
            synapse.job.reason = 'Validator prepares for update'
            return synapse
        
        # Choose un-benchmarked miner
        miner = choose_miner()
        
        if miner is None:
            bt.logging.info(f"No miner to benchmark")
            synapse.miner_uid = -1
            return synapse
        else:
            bt.logging.info(f"Benchmarking Miner UID: {miner['uid']}, Tried: {miner.get('retry', 0)}")
        
        miner['status'] = 'benchmarking'
        miner['timestamp'] = time.time()
        
        # Set miner_uid
        synapse.miner_uid = miner['uid']
        
        # Query the miner status and create job for the bot
        try:
            synapse.job.rank = 1
            synapse.job.world_size = 3
            if hotkey in processes and processes[hotkey]['process'].is_alive():
                bt.logging.warning(f'Master process running {synapse.miner_uid} {hotkey}')
                bt.logging.trace(traceback.format_exc())
                synapse.job.status = 'error'
                synapse.job.reason = 'Master process running'
                return synapse
            # try to join the group
            synapse.job.master_hotkey = axon.wallet.hotkey.ss58_address
            synapse.job.client_hotkey = hotkey
            synapse.job.master_addr = axon.external_ip
            synapse.job.master_port = utils.get_unused_port(9000, 9300)
            synapse.job.session_time = 30
            synapse.job.bandwidth = 5 * 1024 * 1024 # 5 MB
            
            bt.logging.info("⌛ Starting benchmarking process")
            bt.logging.trace(synapse.job)
            
            # input, output queues for communication between the main process and dist_validator process
            input_queue = mp.Queue()
            output_queue = mp.Queue()
            synapse.job.status = "init"
            miners = [(synapse.miner_uid, metagraph.axons[synapse.miner_uid])]
            process = mp.Process(target=start_master_process, args=(input_queue, output_queue, wallet, miners, synapse.job, True))
            process.start()
            job : protocol.Job = output_queue.get()
            bt.logging.info("Master process running")
            # store process information
            processes[hotkey] = {
                'process': process,
                'input_queue': input_queue,
                'output_queue': output_queue,
                'type': 'master',
                'job': job,
                'benchmarking': True,
                'miners': miners
            }
            miner['process'] = process
            miner['input'] = input_queue
            miner['output'] = output_queue
            
            synapse.job = processes[hotkey]['job']
            synapse.job.rank = 1
            bt.logging.trace(synapse.job)
            miner['retry'] = miner.get('retry', 0) + 1
            # create thread for waiting benchmark result
            return synapse
        except Exception as e:
            # if failed, set joining to false
            bt.logging.error(f"🛑 {e}")
            traceback.print_exc()
            synapse.job.reason = str(e)
            del processes[hotkey]
            miner['status'] = 'available'
            return synapse

    def blacklist_request_benchmark(synapse: protocol.RequestBenchmark) -> Tuple[bool, str]:
        hotkey = synapse.dendrite.hotkey
        # Check if the hotkey is benchmark_hotkey
        if hotkey not in validator_config['benchmark_hotkeys']:
            bt.logging.error(f"Hotkey {hotkey} is not benchmark_hotkey")
            synapse.job.reason = f"Hotkey {hotkey} is not benchmark_hotkey"
            return True, ""
        return False, ""

    def connect_master( synapse: protocol.ConnectMaster ) -> protocol.ConnectMaster:
        hotkey = synapse.dendrite.hotkey
        bt.logging.info(f"Connecting request from {hotkey}")
        
        # Version checking
        if not utils.check_version(synapse.version):
            synapse.version = utils.get_my_version()
            return synapse
        
        synapse.version = utils.get_my_version()
        
        if utils.update_flag:
            synapse.job.status = 'error'
            synapse.job.reason = 'Validator prepares for update'
            return synapse
        
        try:
            rank = synapse.job.rank
            if synapse.job.rank == 1:
                if hotkey in processes and processes[hotkey]['process'].is_alive():
                    traceback.print_exc()
                    synapse.job.status = 'error'
                    synapse.job.reason = 'Master process running'
                    return synapse
                # try to join the group
                synapse.job.master_hotkey = axon.wallet.hotkey.ss58_address
                synapse.job.client_hotkey = hotkey
                synapse.job.master_addr = axon.external_ip
                synapse.job.master_port = utils.get_unused_port(9000, 9300)
                
                bt.logging.info("⌛ Starting work process")
                bt.logging.trace(synapse.job)
                
                input_queue = mp.Queue()
                output_queue = mp.Queue()
                synapse.job.status = "init"
                
                # Choose available miners to join.
                miners = [(int(uid), axon) for uid, axon, miner in zip(metagraph.uids, metagraph.axons, miner_status) if axon.is_serving and ((miner['status'] == 'available' and utils.calc_bandwidth_from_memory(miner.get('free_memory',0)) >= synapse.job.bandwidth) or (miner['status'] == 'benchmarked' and utils.calc_bandwidth_from_memory(miner.get('free_memory',0)) >= synapse.job.bandwidth))]
                # sort miners by score
                miners = sorted(miners, key=lambda x: scores[x[0]], reverse=True)
                bt.logging.info(f"Available miners to join: {miners}")
                print(wallet, miners, synapse.job)
                process = mp.Process(target=start_master_process, args=(input_queue, output_queue, wallet, miners, synapse.job, False))
                process.start()
                job : protocol.Job = output_queue.get()
                bt.logging.info("Master process running")
                processes[hotkey] = {
                    'process': process,
                    'input_queue': input_queue,
                    'output_queue': output_queue,
                    'type': 'master',
                    'job': job,
                    'benchmarking': False,
                    'retry': 0,
                    'miners': job.miners
                }
                bt.logging.info(f"Connected miners: {job.miners}")
                for (uid, _) in job.miners:
                    miner_status[uid]['status'] = 'working'
            else:
                if hotkey not in processes:
                    wait_for_master_process(hotkey)
            synapse.job = processes[hotkey]['job']
            synapse.job.rank = rank
            bt.logging.trace(synapse.job)
            return synapse
        except Exception as e:
            # if failed, set joining to false
            bt.logging.warning(f"🛑 {e}")
            bt.logging.trace(traceback.format_exc())
            if hotkey in processes and processes[hotkey]['job']:
                for uid in processes[hotkey]['job'].miners:
                    miner_status[uid]['status'] = 'available'
            synapse.job.status = 'error'
            synapse.job.reason = str(e)
            return synapse
    
    # Prepare benchmark result, benchmark bot information 
    def get_benchmark_result( synapse: protocol.BenchmarkResults) -> protocol.BenchmarkResults:
        
        global miner_status
        
        hotkey = synapse.dendrite.hotkey
        
        bt.logging.info(f"Get benchmark result request from {hotkey}")
        # Version checking
        if not utils.check_version(synapse.version):
            synapse.version = utils.get_my_version()
            bt.logging.error(f"Benchmark Results: Version mismatch {synapse.version}")
            return synapse
        bt.logging.info(f"benchmark results: {synapse}")
        synapse.results = []
        for miner in miner_status:
            synapse.results.append({
                'bandwidth': miner['bandwidth'],
                'speed': miner['speed'],
                'free_memory': miner['free_memory'],
                'upload': miner['upload'],
                'download': miner['download'],
                'url': speedtest_results.get(metagraph.axons[miner['uid']].ip, {}).get('url', ''),
            })
        synapse.bots = []
        bt.logging.info(f"Benchmark results: {hotkey} {synapse.results}")
        return synapse

    def blacklist_get_benchmark_result( synapse: protocol.BenchmarkResults ) -> Tuple[bool, str]:
        hotkey = synapse.dendrite.hotkey
        allowed_hotkeys = [
            "5DkRd1V7eurDpgKbiJt7YeJzQvxgpPiiU6FMDf8RmYQ78DpD", # Allow subnet owner's hotkey to fetch benchmark results from validators
        ]
        # Check if the hotkey is allowed list
        if hotkey not in allowed_hotkeys:
            return True, ""
        return False, ""

    def get_miner_status( synapse: protocol.MinerStatus ) -> protocol.MinerStatus:
        if not utils.check_version(synapse.version):
            synapse.version = utils.get_my_version()
            return synapse
        synapse.version = utils.get_my_version()
        synapse.available = False
        return synapse

    def speedtest():
        
        global speedtest_results
        global miner_status
        # choose axons for speed test
        axons_for_speedtest = []
        for uid, axon in enumerate(metagraph.axons):
            if miner_status[uid] and miner_status[uid]['status'] != 'available':
                continue
            old_speedtest_result = speedtest_results.get(axon.ip, None)
            if old_speedtest_result is None:
                axons_for_speedtest.append((uid, axon))
                continue
            # Speedtest every 72 hours
            if time.time() - old_speedtest_result['timestamp'] > 3600 * 72:
                axons_for_speedtest.append((uid, axon))
                continue
        bt.logging.info(f"🔵 UIDs for Speed Test: { [uid for uid, axon in axons_for_speedtest]}")
        if len(axons_for_speedtest) == 0:
            return
        responses = dendrite.query([axon for uid, axon in axons_for_speedtest], protocol.SpeedTest(version = utils.get_my_version()), timeout = 70)
        bt.logging.success("Got speedtest results")
        current_timestamp = time.time()
        for response, (uid, axon) in zip(responses, axons_for_speedtest):
            if response.result is not None:
                # Convert datetime object to Unix timestamp
                date_time = datetime.fromisoformat(response.result['timestamp'].rstrip("Z")).replace(tzinfo=timezone.utc)
                timestamp = int(date_time.timestamp())
                time.sleep(6)
                
                # Verify speedtest result
                verify_data = verify_speedtest_result(response.result['result']['url'])
                
                if verify_data is None:
                    bt.logging.error(f"Miner {uid}: Failed to verify speedtest result")
                    continue

                if abs(timestamp - verify_data['result']['date']) > 10:
                    bt.logging.error(f"Miner {uid}: Timestamp mismatch {verify_data['result']['date']} {timestamp} ({response.result['timestamp']})")
                    continue
                
                if verify_data['result']['date'] < current_timestamp - 40:                    
                    bt.logging.error(f"Miner {uid}: Speedtest timestamp is too old {verify_data['result']['date']}, current: {current_timestamp}")
                    continue
                
                # if timestamp < timestamp - 40:                    
                #     bt.logging.error(f"Miner {miner['uid']}: Speedtest timestamp is too old {timestamp}")
                #     continue
                
                # miner['upload'] = response.result['upload']['bandwidth'] * 8 / 1000000
                # miner['download'] = response.result['download']['bandwidth'] * 8 / 1000000
                # miner['ping'] = response.result['ping']['latency']
                speedtest_result = {
                    'timestamp': timestamp,
                    'url': response.result['result']['url'],
                    'isp': response.result['isp'],
                    'server_id': response.result['server']['id'],
                    'external_ip': response.result['interface']['externalIp'],
                    'upload': verify_data['result']['upload'],
                    'download': verify_data['result']['download'],
                    'ping': verify_data['result']['latency'],
                }
                speedtest_results[axon.ip] = speedtest_result
                
                bt.logging.success(f"Miner {uid} | Download: {speedtest_result['download']/1000}/Mbps | Upload: {speedtest_result['upload']/1000}/Mbps")
                
                # save speedtest result
                with open('speedtest_results.json', 'w') as f:
                    json.dump(speedtest_results, f, indent=2)
                
        for miner, axon in zip(miner_status, metagraph.axons):
            speedtest_result = speedtest_results.get(axon.ip, None)
            if speedtest_result is None or time.time() - speedtest_result['timestamp'] > 3600 * 72:
                miner['upload'] = 0
                miner['download'] = 0
                continue
            miner['upload'] = speedtest_result['upload']
            miner['download'] = speedtest_result['download']
            miner['ping'] = speedtest_result['ping']
        
        save_miner_status()
        bt.logging.success("✅ Speedtest completed")
    
    def save_miner_status():
        global miner_status
        json_data = [{k: v for k, v in miner.items() if k not in ['process', 'output', 'input']} for miner in miner_status]
        with open('miner_status.json', 'w') as f:
            json.dump(json_data, f, indent=2)
    
    def print_miner_status():
        table = Table(title="Miners")
        table.add_column("uid", justify="right", style="cyan", no_wrap=True)
        table.add_column("upload", style="magenta", min_width=7)
        table.add_column("download", style="magenta", min_width=7)
        table.add_column("speed", style="magenta")
        table.add_column("memory", style="magenta")
        table.add_column("try", style="magenta")
        table.add_column("score", style="magenta")
        table.add_column("status", style="magenta", no_wrap=True)
        for miner in miner_status:
            table.add_row(
                str(miner['uid']),
                f"{(miner['upload']) / 1000:.2f}",
                f"{(miner['download']) / 1000:.2f}",
                f"{(miner['speed']) / 1024 / 1024:.2f}",
                f"{(miner['free_memory']) / 1024 / 1024 / 1024:.2f}",
                f"{miner.get('retry', 0)}",
                f"{miner.get('score', 0) * 100:.2f}",
                status_with_color(miner['status'])
            )
        console = Console()
        console.print(table)
    
    def load_miner_status():
        global miner_status
        try:
            with open('miner_status.json') as f:
                miner_status = json.load(f)
                for miner in miner_status:
                    if miner['status'] == 'benchmarking' or miner['status'] == 'working':
                        miner['status'] = 'available'
                bt.logging.info(f"Loaded miner status from save file")
                for uid, hotkey in enumerate(metagraph.hotkeys):
                    if uid >= len(miner_status):
                        miner_status.append({
                            'uid': uid,
                            'hotkey': hotkey,
                            'free_memory': 0,
                            'bandwidth': 0,
                            'speed': 0,
                            'status': 'unavailable',
                            'timestamp': 0,
                            'retry': 0,
                            'upload': 0,
                            'download': 0,
                        })
                    if miner_status[uid]['hotkey'] != hotkey:
                        miner_status[uid] = {
                            'uid': uid,
                            'hotkey': hotkey,
                            'free_memory': 0,
                            'bandwidth': 0,
                            'speed': 0,
                            'status': 'unavailable',
                            'timestamp': 0,
                            'retry': 0,
                            'upload': 0,
                            'download': 0,
                        }
                print_miner_status()
        except:
            init_miner_status()
    
    def status_with_color(status):
        icon = '⚫'
        if status == 'benchmarked':
            icon = '🟢'
        if status == 'available' or status == 'benchmarking' or status == 'working':
            icon = '🔵'
        if status == 'failed':
            icon = '🛑'
        return f"{icon} {status}"
    
    # load speedtest results
    try:
        with open('speedtest_results.json') as f:
            speedtest_results = json.load(f)
            bt.logging.info(f"Loaded speedtest results from save file: {json.dumps(speedtest_results, indent=2)}")
    except:
        pass
 
    # load miner status
    load_miner_status()
 
    # Attach determiners which functions are called when servicing a request.
    bt.logging.info(f"Attaching forward function to axon.")
    axon.attach(
        forward_fn = connect_master,   
    ).attach(
        forward_fn = request_benchmark,
        blacklist_fn = blacklist_request_benchmark
    ).attach(
        forward_fn = get_benchmark_result,
        blacklist_fn = blacklist_get_benchmark_result
    ).attach(
        forward_fn = get_miner_status,
    )

    # Serve passes the axon information to the network + netuid we are hosting on.
    # This will auto-update if the axon port of external ip have changed.
    bt.logging.info(f"Serving axon on network: {config.subtensor.chain_endpoint} with netuid: {config.netuid}")
    axon.serve( netuid = config.netuid, subtensor = subtensor )

    # Start  starts the miner's axon, making it active on the network.
    bt.logging.info(f"Starting axon server on port: {config.axon.port}")
    axon.start()

    # Check processes
    thread = Thread(target=utils.check_processes, args=(processes, miner_status))
    thread.start()
    
    # Check benchmark processes
    benchmark_thread = Thread(target=check_benchmark_result)
    benchmark_thread.start()

    # Step 6: Keep the validator alive
    # This loop maintains the validator's operations until intentionally stopped.
    bt.logging.info(f"Starting main loop")
    step = 0

    scores = torch.ones_like(metagraph.S, dtype=torch.float32)
    bt.logging.info(f"Weights: {scores}")
    
    scores_file = "scores.pt"
    try:
        scores = torch.load(scores_file)
        bt.logging.info(f"Loaded scores from save file: {scores}")
    except:
        scores = torch.zeros_like(metagraph.S, dtype=torch.float32)
        bt.logging.info(f"Initialized all scores to 0")
    
    # set all nodes without ips set to 0
    scores = scores * torch.Tensor([metagraph.neurons[uid].axon_info.ip != '0.0.0.0' for uid in metagraph.uids])
    step = 0

    alpha = 0.8
    
    global last_benchmark_at
    
    last_benchmark_at = time.time()
    while True:
        try:
            # Below: Periodically update our knowledge of the network graph.
            metagraph = subtensor.metagraph(config.netuid)
            last_updated_block = metagraph.last_update[my_subnet_uid].item()
            bt.logging.info(f"Last benchmarked at: {last_benchmark_at}")
            if last_benchmark_at > 0 and time.time() - last_benchmark_at > 300:
                bt.logging.error("No benchmark is happening. Restarting validator ...")
                time.sleep(1)
                axon.stop()
                os._exit(0)
                
            if step % 5 == 0:
                update_miner_status()
                print_miner_status()
                # Speed Test
                speedtest()
                
            # Periodically update the weights on the Bittensor blockchain.
            current_block = subtensor.block
            bt.logging.info(f"Last updated block: {last_updated_block}, current block: {current_block}")
                
            if current_block - last_updated_block > 200 or (config.netuid == 32 and current_block - last_updated_block > 10):
                
                # Skip setting weight if there are miners benchmarking or not benchmarked yet
                is_benchmarking = False
                for miner in miner_status:
                    if miner['status'] == 'benchmarking' or miner['status'] == 'available':
                        bt.logging.debug("Benchmarking is in progress, skip score calculation")
                        is_benchmarking = True
                        break
                        
                if is_benchmarking:
                    if current_block - last_updated_block < 400:
                        step += 1
                        time.sleep(bt.__blocktime__)
                        continue
                    else: 
                        is_benchmarking = False
                        for miner in miner_status:
                            if miner['status'] == 'benchmarked':
                                is_benchmarking = True
                                break
                        if not is_benchmarking and step > 50:
                            bt.logging.error("No miner is benchmarked, something wrong")
                            bt.logging.info("Restarting validator ...")
                            time.sleep(2)
                            axon.stop()
                            os._exit(0)
                
                bt.logging.success("Updating score ...")
                        
                new_scores = calculate_score()
                
                scores = new_scores * alpha + scores * (1 - alpha)
                
                for uid in range(len(metagraph.uids)):
                    miner_status[uid]['new_score'] = float(new_scores[uid])
                    miner_status[uid]['score'] = float(scores[uid])
                
                print_miner_status()
                
                weights = torch.nn.functional.normalize(scores, p=1.0, dim=0)
                
                ( processed_uids, processed_weights,) = bt.utils.weight_utils.process_weights_for_netuid(
                    uids=metagraph.uids,
                    weights=weights,
                    netuid = config.netuid,
                    subtensor=subtensor
                )
                
                table = Table(title="Weights")
                table.add_column("uid", justify="right", style="cyan", no_wrap=True)
                table.add_column("weight", style="magenta")
                for index, weight in list(zip(processed_uids.tolist(), processed_weights.tolist())):
                    table.add_row(str(index), str(round(weight, 4)))
                console = Console()
                console.print(table)
                
                # This is a crucial step that updates the incentive mechanism on the Bittensor blockchain.
                # Miners with higher scores (or weights) receive a larger share of TAO rewards on this subnet.
                result = subtensor.set_weights(
                    netuid = config.netuid, # Subnet to set weights on.
                    wallet = wallet, # Wallet to sign set weights using hotkey.
                    uids = processed_uids, # Uids of the miners to set weights for.
                    weights = processed_weights, # Weights to set for the miners. 
                    wait_for_inclusion=True,
                )
                
                if result: 
                    bt.logging.success('✅ Successfully set weights.')
                    torch.save(scores, scores_file)
                    bt.logging.info(f"Saved weights to \"{scores_file}\"")
                    last_updated_block = current_block
                    init_miner_status()
                    
                else: bt.logging.error('Failed to set weights.')
                
            # Check for auto update
            if step % 5 == 0 and config.auto_update != "no":
                utils.update_repository()
            
            step += 1
            time.sleep(bt.__blocktime__)


        # If someone intentionally stops the miner, it'll safely terminate operations.
        except KeyboardInterrupt:
            axon.stop()
            bt.logging.success('Validator killed by keyboard interrupt.')
            break
        # In case of unforeseen errors, the miner will log the error and continue operations.
        except Exception as e:
            bt.logging.error(traceback.format_exc())
            continue

def wait_for_master_process(hotkey, timeout=20):
    start_time = time.time()
    while(True):
        time.sleep(0.1) 
        if hotkey in processes:
            break
        if time.time() - start_time > timeout:
            raise TimeoutError('Timeout while waiting for master process')

# This is the main function, which runs the miner.
if __name__ == "__main__":
    # mp.set_start_method('spawn')  # This is often necessary in PyTorch multiprocessing
    main( get_config() )
