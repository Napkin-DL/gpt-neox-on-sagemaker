# Copyright (c) 2021, EleutherAI contributors
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pretrain"""
from megatron.neox_arguments import NeoXArgs
from megatron.training import pretrain

### ADDed for SageMaker
from megatron.utils import get_wandb_api_key

if __name__ == "__main__":
    
    ### ADDed for SageMaker
    neox_args = NeoXArgs.consume_deepy_args()
    deepspeed_main_args = neox_args.get_deepspeed_main_args()

    # Extract wandb API key and inject into worker environments
    wandb_token = get_wandb_api_key(neox_args=neox_args)
    if wandb_token is not None:
        deepspeed.launcher.runner.EXPORT_ENVS.append("WANDB_API_KEY")
        os.environ["WANDB_API_KEY"] = wandb_token
    neox_args = NeoXArgs.consume_neox_args(overwrite_values=deepspeed_main_args[-1])    
    ### ADDed for SageMaker
#     neox_args = NeoXArgs.consume_neox_args()
    
    neox_args.configure_distributed_args()
    neox_args.build_tokenizer()  # tokenizer needs to be build in training in order to set the padding vocab
    neox_args.initialize_tensorboard_writer()  # is initialized if tensorboard directory is defined
    pretrain(neox_args=neox_args)
