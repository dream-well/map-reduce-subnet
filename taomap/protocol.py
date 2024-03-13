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

from typing import Optional, List, Dict
import bittensor as bt
import pydantic
import time

"""
Extends the Bittensor Synapse with an additional version attribute, 
used for compatibility and version control in mapreduce operations.
"""
class MapSynapse ( bt.Synapse ):
    version: Optional[str] = None

"""
Defines the structure of a mapreduce job, including information about network configuration, 
the participating miners, and the job's runtime status.
"""
class Job (pydantic.BaseModel):
    master_hotkey: Optional[str] = None
    client_hotkey: Optional[str] = None
    master_addr: Optional[str] = None
    master_port: Optional[int] = None
    world_size: Optional[int] = None
    rank: Optional[int] = None
    peer_count: Optional[int] = None
    miners: Optional[List] = []
    verifier_count: Optional[int] = None
    bandwidth: Optional[int] = None
    started_at: Optional[int] = None
    session_time: Optional[int] = 300
    status: Optional[str] = None
    reason: Optional[str] = None

"""
A specialized Synapse representing the status of a miner, 
including its availability and memory resources.
"""
class MinerStatus( MapSynapse ):
    job_id: Optional[int] = None

    def deserialize(self):
        if self.job_id is None:
            self.job_id = -1
        status = ""
        if self.job_id == -1:
            status = "offline"
        elif self.job_id == 0:
            status = "idle"
        elif self.job_id == 1:
            status = "benchmarking"
        elif self.job_id >= 2:
            status = f"working"
        return self.job_id, status

"""
Speed test

TODO: Add security checks
"""
class Benchmark( MapSynapse ):
    shape: Optional[List[int]] = None
    tensor: Optional[bt.Tensor] = None
    
    def deserialize(self):
        tensor = self.tensor.deserialize()
        size_in_bytes = tensor.element_size() * tensor.numel()
        return [time.time(), size_in_bytes]

class ShareGradients():
    gradiens: Optional[List[bt.Tensor]]