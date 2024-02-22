# utils.py
# Utility Functions and Classes for MapReduce Operations
# The `utils.py` module in this MapReduce application contains a series of helper functions and classes that provide common, reusable functionalities required by different components of the application. These utilities streamline tasks such as data formatting, network communication, version checking, and memory management. 
# The use of a dedicated utility 
# module promotes code reusability and modularity, enabling a cleaner, more maintainable, and efficient codebase. Functions in this module are designed to be generic and abstract enough to be used in various parts of the application without modification, illustrating a key principle of DRY (Don't Repeat Yourself).

import bittensor as bt
import os
import time
import mapreduce
import netifaces
import torch
import socket
import psutil
import random
import codecs
import re
import traceback
from datetime import datetime
import subprocess

update_flag = False
update_at = 0

"""
Converts a size in bytes to a more human-readable format (e.g., KB, MB).

Args:
    size (int): The size in bytes.
    decimal_places (int, optional): Number of decimal places for the size. Defaults to 2.

Returns:
    str: Human-readable size string.
"""
def human_readable_size(size, decimal_places=2):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f} {unit}"

"""
Calculates the size of a specific chunk given the total file size and number of chunks.

Args:
    file_size (int): Total size of the file.
    chunk_count (int): Total number of chunks to divide the file into.
    chunk_id (int): The specific chunk ID for which size is calculated.

Returns:
    int: Size of the specified chunk.
"""
def get_chunk_size(file_size, chunk_count, chunk_id):
    chunk_size = file_size // chunk_count
    if chunk_id == chunk_count - 1:
        chunk_size += file_size % chunk_count
    return chunk_size
    
"""
Splits a tensor into chunks with padding added to the last chunk if needed.

Args:
    tensor (Tensor): The tensor to be chunked.
    num_chunks (int): Number of chunks to split the tensor into.
    dim (int, optional): The dimension along which to split the tensor. Defaults to 0.

Returns:
    List[Tensor]: List of tensor chunks.
"""
def chunk_with_padding(tensor, num_chunks, dim=0):
    # Determine the size of each chunk
    chunk_size = (tensor.size(dim) + num_chunks - 1) // num_chunks  # Ceiling division

    # Split the tensor into chunks
    chunks = list(tensor.chunk(num_chunks, dim=dim))  # Convert to list for mutability
    
    # Pad the last chunk if necessary
    last_chunk = chunks[-1]
    if last_chunk.size(dim) < chunk_size:
        # Calculate the padding size
        pad_size = chunk_size - last_chunk.size(dim)
        
        # Create a padding tensor of zeros
        pad_tensor = torch.zeros(*last_chunk.shape[:dim], pad_size, *last_chunk.shape[dim+1:], device=tensor.device, dtype=tensor.dtype)
        
        # Concatenate the padding tensor to the last chunk
        chunks[-1] = torch.cat([last_chunk, pad_tensor], dim=dim)  # Now this line will work
    
    return chunks  # You can convert it back to a tuple if necessary by `return tuple(chunks)`

"""
Merges a list of chunks back into a single tensor.

Args:
    chunks (List[Tensor]): List of tensor chunks to merge.
    original_size (int): The size of the original tensor before chunking.
    dim (int, optional): The dimension along which to merge the chunks. Defaults to 0.

Returns:
    Tensor: The merged tensor.
"""
def merge_chunks(chunks, original_size, dim=0):
    # Concatenate the chunks along the given dimension
    merged_tensor = torch.cat(chunks, dim=dim)
    # Trim the tensor to the original size
    merged_tensor = merged_tensor.narrow(dim, 0, original_size)
    return merged_tensor

"""
Retrieves the current version of the MapReduce protocol being used.

Returns:
    mapreduce.protocol.Version: The version object with major, minor, and patch components.
"""
def get_my_version() -> mapreduce.protocol.Version:
    version_str = mapreduce.__version__
    major, minor, patch = version_str.split('.')
    return mapreduce.protocol.Version(
        major_version = int(major),
        minor_version = int(minor),
        patch_version = int(patch)
    )


def timestamp_to_datestring(timestamp):
    # Convert the timestamp to a datetime object
    dt_object = datetime.fromtimestamp(timestamp)

    # Format the datetime object as an ISO 8601 string
    iso_date_string = dt_object.isoformat()
    return iso_date_string

def set_update_flag():
    global update_flag
    global update_at
    if update_flag:
        bt.logging.info(f"🧭 Auto Update scheduled on {timestamp_to_datestring(update_at)}")
        return
    update_flag = True
    update_at = time.time() + 1800
    bt.logging.info(f"🧭 Auto Update scheduled on {timestamp_to_datestring(update_at)}")

"""
Checks if the provided version matches the current MapReduce protocol version.

Args:
    version (mapreduce.protocol.Version): The version to check.
    flag: major | minor | patch | no

Returns:
    bool: True if the versions match, False otherwise.
"""
def check_version( version: mapreduce.protocol.Version ) -> bool:
    global update_flag
    version_str = mapreduce.__version__
    major, minor, patch = version_str.split('.')
    other_version_str = f"{version.major_version}.{version.minor_version}.{version.patch_version}"
    if version.major_version != int(major):
        bt.logging.error("🔴 Major version mismatch", f"yours: {version_str}, other's: {other_version_str}")
        return False
    elif version.minor_version != int(minor):
        bt.logging.warning("🟡 Minor version mismatch", f"yours: {version_str}, other's: {other_version_str}")
    elif version.patch_version != int(patch):
        bt.logging.warning("🔵 Patch version mismatch", f"yours: {version_str}, other's: {other_version_str}")
    return True

"""
Checks the status of running processes and updates their status accordingly.

Args:
    processes (dict): Dictionary containing process information.
    miner_status (dict, optional): Dictionary containing the status of miners.
"""
def check_processes(processes, miner_status = None):
    global update_flag
    while True:
        try:
            keys_to_delete = []
            all_keys = list(processes.keys())
            for key in all_keys:
                process = processes[key]['process']
                if not process.is_alive():
                    keys_to_delete.append(key)
            for key in keys_to_delete:
                bt.logging.info(f"🚩 Delete Process: {key}")
                if 'miners' in processes[key]:
                    for (uid, _) in processes[key]['miners']:
                        if miner_status[int(uid)]['status'] == 'working':
                            miner_status[int(uid)]['status'] = 'available'
                del processes[key]
            # Check if upgrade is needed
            if update_flag:
                if len(processes) == 0:
                    update_repository()
                    bt.logging.info("🔁 Exiting process for update.")
                    print("\033[92m🔁 Exiting process for update.\033[0m")
                    os._exit(0)
                else:
                    bt.logging.info(f"⌛️ Waiting for {len(processes)} processes to finish.")
                if time.time() > update_at:
                    update_repository()
                    bt.logging.warning("\033[93m🔁 Force exiting process for update.\033[0m")
                    os._exit(0)
        except Exception as e:
            bt.logging.warning(f"Error checking processes: {e}")
            bt.logging.trace(traceback.format_exc())
        time.sleep(0.005)
        
"""
Checks if any process in the given dictionary is still alive.

Args:
    processes (dict): Dictionary containing process information.

Returns:
    bool: True if any process is alive, False otherwise.
"""
def is_process_running(processes):
    for key in processes:
        process = processes[key]['process']
        if process.is_alive():
            return True
    return False

"""
Finds the network interface associated with a given IP address.

Args:
    target_ip (str): The target IP address.

Returns:
    str: The name of the network interface.

Raises:
    ValueError: If no interface is found for the given IP.
"""
def find_network_interface(target_ip):
    for interface in netifaces.interfaces():
        addrs = netifaces.ifaddresses(interface)
        if netifaces.AF_INET in addrs:
            for addr_info in addrs[netifaces.AF_INET]:
                if addr_info['addr'] == target_ip:
                    return interface
    # raise ValueError(f"No interface found for IP {target_ip}")
    bt.logging.warning(f"No interface found for IP {target_ip}")
    return None

def find_default_interface():
    # Get the default gateway details
    gws = netifaces.gateways()
    default_gateway = gws['default'][netifaces.AF_INET]  # AF_INET for IPv4
    interface = default_gateway[1]
    bt.logging.info(f"Default internet interface: {interface}")
    # Optionally, get the IP address of the default interface
    addrs = netifaces.ifaddresses(interface)
    ip_info = addrs[netifaces.AF_INET][0]
    ip_address = ip_info['addr']
    bt.logging.info(f"IP Address of default interface: {ip_address}")
    return interface

"""
Sets the network interface for Gloo (used in distributed operations) based on the external IP.

Args:
    external_ip (str): The external IP address.
"""
def set_gloo_socket_ifname(external_ip):
    ifname = find_network_interface(external_ip)
    bt.logging.info(f"IP: {external_ip} IFNAME: {ifname}")
    if ifname is not None:
        os.environ['GLOO_SOCKET_IFNAME'] = ifname
    else:
        os.environ['GLOO_SOCKET_IFNAME'] = find_default_interface()

"""
Finds an unused port within a specified range.

Args:
    start_port (int): The starting port number.
    end_port (int): The ending port number.

Returns:
    int: An unused port number within the specified range.

Raises:
    Exception: If no unused port is found.
"""
def get_unused_port(start_port, end_port):
    """
    This function finds unused ports in the given range.

    Args:
    start_port (int): The starting port number.
    end_port (int): The ending port number.

    Returns:
    List[int]: A list of unused ports within the given range.
    """
    unused_ports = []
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            res = s.connect_ex(('localhost', port))
            if res != 0:
                unused_ports.append(port)
    if len(unused_ports) == 0:
        raise Exception("No unused ports found")
    #  select random port in unused ports
    return unused_ports[random.randint(0, len(unused_ports) - 1)]


"""
Retrieves the amount of available bandwidth from the free memory.
"""

def calc_bandwidth_from_memory(free_memory: int):
    return max(int((free_memory - 500 * 1024 * 1024) / 2), 0)

"""
Retrieves the amount of free memory in the system.

Returns:
    int: Free memory in bytes.
"""
def get_available_memory():
    # Get the memory details
    memory = psutil.virtual_memory()
    # Free memory in bytes
    return memory.available

'''
Check if the repository is up to date
'''
def update_repository():
    bt.logging.info("Updating repository")
    os.system("git pull")
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, '__init__.py'), encoding='utf-8') as init_file:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", init_file.read(), re.M)
        new_version = version_match.group(1)
        bt.logging.success(f"local version: {mapreduce.__version__}, remote version: {new_version}")
        if mapreduce.__version__ != new_version:
            os.system("python3 -m pip uninstall mapreduce -y")
            os.system("python3 -m pip uninstall map-reduce-subnet -y")
            os.system("python3 -m pip install -e .")
            set_update_flag()
            return True
    return False


def generate_perf_app(secret_key):
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        script_name = os.path.join(here, 'performance.py')
    
        # Read the content of the script.py file
        with open(script_name, 'r') as file:
            script_content = file.read()
    
        # Find and replace the script_key value
    
        pattern = r"secret_key\s*=\s*.*?#key"
        script_content = re.sub(pattern, f"secret_key = {secret_key}#key", script_content, count=1)
    
        # Write the modified content back to the file
        with open(script_name, 'w') as file:
            file.write(script_content)
    
        # Run the pyinstaller command
        command = f'cd {here}\npyinstaller --onefile performance.py'
        try:
            subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            bt.logging.error("An error occurred while generating the app.")
            bt.logging.error(f"Error output:{e.stderr.decode()}")
    except Exception as e:
        bt.logging.error(f"{e}")
