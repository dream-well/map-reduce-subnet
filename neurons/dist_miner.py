from datetime import timedelta
import torch
import torch.distributed as dist
from threading import Lock, Thread
from argparse import ArgumentParser
from mapreduce.utils import human_readable_size, set_gloo_socket_ifname
import torch.multiprocessing as mp
import bittensor as bt
import mapreduce
import os

# This class is responsible for initializing the miner, setting up the process group, initializing groups,
# scattering data to miners, gathering data from miners, reducing data from peers, and monitoring actions for the miner.
# it assumes each miner has a unique rank and uses TCP initialization method
class MinerDist:
    # Initialize the miner with the given IP address
    # This method sets the IP address of the miner and sets the gloo socket interface name to the IP address.
    def __init__(self, ip_address, opened_ports):
        self.ip_address = ip_address
        self.opened_ports = opened_ports
        set_gloo_socket_ifname(self.ip_address)
        os.environ['GLOO_PORT_RANGE'] = opened_ports

    # Initialize the process group for the miner
    # This method sets up the process group for the miner using the job details.
    def initialize_process_group(self, job: mapreduce.protocol.Job):
        bt.logging.info(f"Initializing process group for miner with rank {job.rank}")
        bt.logging.trace(job)
        self.rank = job.rank
        self.peer_count = job.peer_count
        self.world_size = job.world_size
        self.master_addr = job.master_addr
        self.master_port = job.master_port
        if not dist.is_initialized():
            bt.logging.info(f"tcp://{self.master_addr}:{self.master_port}, rank: {self.rank} world_size: {self.world_size}")
            dist.init_process_group(
                init_method=f"tcp://{self.master_addr}:{self.master_port}",
                backend='gloo',
                rank=self.rank,
                world_size=self.world_size,
                timeout=timedelta(seconds=10)
            )
            bt.logging.info(f"Miner (Rank {self.rank}) initialized.")
            # Create groups
            self.initialize_groups()
            self.monitor_actions()
        else:
            bt.logging.info(f"Miner with rank {self.rank} is already initialized.")

    # Initialize the groups for the miner
    # This method sets up the groups for the miner.
    def initialize_groups(self):
        self.peers = dist.new_group(ranks=list(range(1, self.peer_count + 1)))
        self.miners = dist.new_group(ranks=list(range(self.peer_count + 1, self.peer_count + self.peer_count + 1)))
        self.peer_miners = [dist.new_group(ranks=[peer_rank] + list(range(self.peer_count + 1, self.peer_count + self.peer_count + 1))) for peer_rank in range(1, self.peer_count + 1)]
        self.miner_peers = [dist.new_group(ranks=[miner_rank] + list(range(1, self.peer_count + 1))) for miner_rank in range(self.peer_count + 1, self.peer_count + self.peer_count + 1)]
        self.validator_rank1 = dist.new_group(ranks=(0, 1))
        bt.logging.info('Groups initialized')

    # Scatter data to miners
    # This method scatters data to miners.
    def scatter_to_miners(self, source, chunk_shape, data_type):
        self.chunk = torch.empty(chunk_shape, dtype=data_type)
        tensor_size = self.chunk.element_size() * self.chunk.nelement()
        bt.logging.info(f'⌛ Scattering to miner ... \033[92m{human_readable_size(tensor_size)}\033[0m')
        dist.scatter(self.chunk, None, src=source, group=self.peer_miners[source-1])
        bt.logging.info(self.chunk)
        bt.logging.info('✅ Scattered')

    # Gather data from miners
    # This method gathers data from miners.
    def gather_from_miners(self, start = 1):
        # Each peer gathers chunks from all miners
        # async tasks
        bt.logging.info('Gathering ...')
        handles = []
        for i in range(start, self.peer_count + 1):
            handles.append(dist.gather(self.chunk, dst=i, group=self.peer_miners[i-1], async_op=True))
        for handle in handles:
            handle.wait()
        bt.logging.info('✅ All Gathered')

    # Reduce data from peers
    # This method reduces data from peers.
    def reduce_from_peers(self, chunk_shape, data_type):
        # Miner reduces chunks from all peers
        self.chunk = torch.zeros(chunk_shape, dtype=data_type)
        tensor_size = self.chunk.element_size() * self.chunk.nelement()
        
        bt.logging.info(f'Reducing Chunks-{self.rank - self.peer_count}... \033[92m{human_readable_size(tensor_size)}\033[0m')
        
        dist.reduce(self.chunk, op=dist.ReduceOp.SUM , dst=self.rank, group=self.miner_peers[self.rank - self.peer_count - 1])
        # Calculate average
        self.chunk = self.chunk / self.peer_count
        bt.logging.trace(self.chunk)
        bt.logging.info('✅ Reduced')

    # Monitor actions for the miner
    # This method monitors actions for the miner.
    def monitor_actions(self):
        while True:
            bt.logging.info('Waiting for next actions')
            actions = [{}]
            dist.broadcast_object_list(actions, src=0)
            action = actions[0]
            bt.logging.info(f"🔔 Action: {action['type']}")
            bt.logging.trace(action)
            if action['type'] == 'broadcast':
                self.scatter_to_miners(action['src'], action['chunk_shape'], action['dtype'])
                self.gather_from_miners(start=2)
            if action['type'] == 'all_reduce':
                self.reduce_from_peers(action['chunk_shape'], action['dtype'])
                self.gather_from_miners()
            if action['type'] == 'exit':
                bt.logging.success('🟢 Received exit signal')
                dist.destroy_process_group()
                break
            bt.logging.success(f"✅ Action: {action['type']} completed")
                
    # This function is modified to accept an additional `axon_port` parameter
from datetime import timedelta
import torch
import torch.distributed as dist
from threading import Lock, Thread
from argparse import ArgumentParser
from mapreduce.utils import human_readable_size, set_gloo_socket_ifname
import torch.multiprocessing as mp
import bittensor as bt
import mapreduce
import os

# This class is responsible for initializing the miner, setting up the process group, initializing groups,
# scattering data to miners, gathering data from miners, reducing data from peers, and monitoring actions for the miner.
# it assumes each miner has a unique rank and uses TCP initialization method
class MinerDist:
    # Initialize the miner with the given IP address
    # This method sets the IP address of the miner and sets the gloo socket interface name to the IP address.
    def __init__(self, ip_address, opened_ports):
        self.ip_address = ip_address
        self.opened_ports = opened_ports
        set_gloo_socket_ifname(self.ip_address)
        os.environ['GLOO_PORT_RANGE'] = opened_ports

    # Initialize the process group for the miner
    # This method sets up the process group for the miner using the job details.
    def initialize_process_group(self, job: mapreduce.protocol.Job):
        bt.logging.info(f"Initializing process group for miner with rank {job.rank}")
        bt.logging.trace(job)
        self.rank = job.rank
        self.peer_count = job.peer_count
        self.world_size = job.world_size
        self.master_addr = job.master_addr
        self.master_port = job.master_port
        if not dist.is_initialized():
            bt.logging.info(f"tcp://{self.master_addr}:{self.master_port}, rank: {self.rank} world_size: {self.world_size}")
            dist.init_process_group(
                init_method=f"tcp://{self.master_addr}:{self.master_port}",
                backend='gloo',
                rank=self.rank,
                world_size=self.world_size,
                timeout=timedelta(seconds=10)
            )
            bt.logging.info(f"Miner (Rank {self.rank}) initialized.")
            # Create groups
            self.initialize_groups()
            self.monitor_actions()
        else:
            bt.logging.info(f"Miner with rank {self.rank} is already initialized.")

    # Initialize the groups for the miner
    # This method sets up the groups for the miner.
    def initialize_groups(self):
        self.peers = dist.new_group(ranks=list(range(1, self.peer_count + 1)))
        self.miners = dist.new_group(ranks=list(range(self.peer_count + 1, self.peer_count + self.peer_count + 1)))
        self.peer_miners = [dist.new_group(ranks=[peer_rank] + list(range(self.peer_count + 1, self.peer_count + self.peer_count + 1))) for peer_rank in range(1, self.peer_count + 1)]
        self.miner_peers = [dist.new_group(ranks=[miner_rank] + list(range(1, self.peer_count + 1))) for miner_rank in range(self.peer_count + 1, self.peer_count + self.peer_count + 1)]
        self.validator_rank1 = dist.new_group(ranks=(0, 1))
        bt.logging.info('Groups initialized')

    # Scatter data to miners
    # This method scatters data to miners.
    def scatter_to_miners(self, source, chunk_shape, data_type):
        self.chunk = torch.empty(chunk_shape, dtype=data_type)
        tensor_size = self.chunk.element_size() * self.chunk.nelement()
        bt.logging.info(f'⌛ Scattering to miner ... \033[92m{human_readable_size(tensor_size)}\033[0m')
        dist.scatter(self.chunk, None, src=source, group=self.peer_miners[source-1])
        bt.logging.info(self.chunk)
        bt.logging.info('✅ Scattered')

    # Gather data from miners
    # This method gathers data from miners.
    def gather_from_miners(self, start = 1):
        # Each peer gathers chunks from all miners
        # async tasks
        bt.logging.info('Gathering ...')
        handles = []
        for i in range(start, self.peer_count + 1):
            handles.append(dist.gather(self.chunk, dst=i, group=self.peer_miners[i-1], async_op=True))
        for handle in handles:
            handle.wait()
        bt.logging.info('✅ All Gathered')

    # Reduce data from peers
    # This method reduces data from peers.
    def reduce_from_peers(self, chunk_shape, data_type):
        # Miner reduces chunks from all peers
        self.chunk = torch.zeros(chunk_shape, dtype=data_type)
        tensor_size = self.chunk.element_size() * self.chunk.nelement()
        
        bt.logging.info(f'Reducing Chunks-{self.rank - self.peer_count}... \033[92m{human_readable_size(tensor_size)}\033[0m')
        
        dist.reduce(self.chunk, op=dist.ReduceOp.SUM , dst=self.rank, group=self.miner_peers[self.rank - self.peer_count - 1])
        # Calculate average
        self.chunk = self.chunk / self.peer_count
        bt.logging.trace(self.chunk)
        bt.logging.info('✅ Reduced')

    # Monitor actions for the miner
    # This method monitors actions for the miner.
    def monitor_actions(self):
        while True:
            bt.logging.info('Waiting for next actions')
            actions = [{}]
            dist.broadcast_object_list(actions, src=0)
            action = actions[0]
            bt.logging.info(f"🔔 Action: {action['type']}")
            bt.logging.trace(action)
            if action['type'] == 'broadcast':
                self.scatter_to_miners(action['src'], action['chunk_shape'], action['dtype'])
                self.gather_from_miners(start=2)
            if action['type'] == 'all_reduce':
                self.reduce_from_peers(action['chunk_shape'], action['dtype'])
                self.gather_from_miners()
            if action['type'] == 'exit':
                bt.logging.success('🟢 Received exit signal')
                dist.destroy_process_group()
                break
            bt.logging.success(f"✅ Action: {action['type']} completed")
                
# This function is modified to accept an additional `axon_port` parameter
def start_miner_dist_process(queue: mp.Queue, external_ip: str, opened_ports: str, wallet: bt.wallet, job: mapreduce.protocol.Job, axon_port: int):
    # Create a new instance of MinerDist with the given IP address and opened ports
    miner_dist = MinerDist(external_ip, opened_ports)
    
    # Set the axon port in the environment or pass it to where it needs to be used
    os.environ['AXON_PORT'] = str(axon_port)
    
    # Initialize the process group for the miner using the job details
    miner_dist.initialize_process_group(job)
    
    # ... rest of the function remains the same

    
