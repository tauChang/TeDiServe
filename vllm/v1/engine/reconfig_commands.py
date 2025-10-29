import abc
import asyncio
from typing import List, Dict, Optional, Set
from vllm.utils import init_logger
from vllm.v1.executor.executors_manager import ExecutorsManager

logger = init_logger(__name__)

def print_command_tree(cmd: 'ReconfigCommand', level=0):
    logger.debug("  " * level + str(cmd))
    for child in cmd.children:
        print_command_tree(child, level + 1)

class ReconfigCommand:
    def __init__(self,
                 executor_id: int,
                 parents: Set['ReconfigCommand'],
                 children: Set['ReconfigCommand']):
        self.executor_id = executor_id
        self.parents = parents # commands that this command depends on
        self.children = children # commands that depends on this command

    async def _execute(self, executors_manager: ExecutorsManager):
        pass

        
    async def execute(self, executors_manager: ExecutorsManager):    
        logger.debug(f"Executing command on executor {self.executor_id} with {len(self.parents)} parents and {len(self.children)} children")
        assert len(self.parents) == 0
        
        # execute
        await self._execute(executors_manager)
        
        # notify and execute children
        tasks = []
        for child in self.children:
            logger.debug(f"Removing parent from child {child}")
            child.parents.remove(self)
            if len(child.parents) == 0:
                logger.debug(f"Executing child {child}")
                tasks.append(asyncio.create_task(
                    child.execute(executors_manager)))
        
        if len(tasks) > 0:
            await asyncio.gather(*tasks)
        
        logger.debug(f"Finished command on executor {self.executor_id}")
        

    
class KillCommand(ReconfigCommand):
    def __init__(self, 
                 executor_id: int,
                 parents: Set['ReconfigCommand'],
                 children: Set['ReconfigCommand'],
                 ):
        super().__init__(executor_id, parents, children)
    
    def __str__(self):
        return f"KillCommand(executor_id={self.executor_id}, parents={len(self.parents)}, children={len(self.children)})"
    
    async def _execute(self, 
                       executors_manager: ExecutorsManager):
        logger.debug(f"Executing Kill command on executor {self.executor_id}")
        await executors_manager.kill_executor(self.executor_id)
        logger.debug(f"Finished Kill command on executor {self.executor_id}")
        
class LaunchCommand(ReconfigCommand):
    def __init__(
                self, 
                executor_id: int,
                bundle_ids: List[int],
                parents: Set['ReconfigCommand'],
                children: Set['ReconfigCommand'],
            ):
        super().__init__(executor_id, parents, children)
        self.bundle_ids = bundle_ids
    
    def __str__(self):
        return f"LaunchCommand(executor_id={self.executor_id}, bundle_ids={self.bundle_ids}, parents={len(self.parents)}, children={len(self.children)})"
    
    async def _execute(self, 
                      executors_manager: ExecutorsManager):
        logger.debug(f"Executing Launch command on executor {self.executor_id}")
        await executors_manager.launch_executor(
            self.executor_id, self.bundle_ids)
        logger.debug(f"Finished Launch command on executor {self.executor_id}")

# class SequentialCommands(ReconfigCommand):
#     def __init__(self, commands: List[ReconfigCommand]):
#         self.commands = commands
    
#     async def execute(self, 
#                       executors: Dict[int, Executor], 
#                       executor_class, 
#                       vllm_config):
#         logger.debug(f"Executing SequentialCommands with {len(self.commands)} commands")
#         for i, cmd in enumerate(self.commands):
#             logger.debug(f"Executing command {i+1}/{len(self.commands)}")
#             await cmd.execute(executors, executor_class, vllm_config)


# class ParallelCommands(ReconfigCommand):
#     def __init__(self, commands: List[ReconfigCommand]):
#         self.commands = commands
    
#     async def execute(self, 
#                       executors: Dict[int, Executor], 
#                       executor_class, 
#                       vllm_config):
#         logger.debug(f"Executing ParallelCommands with {len(self.commands)} commands")
#         await asyncio.gather(
#             *[cmd.execute(executors, executor_class, vllm_config) \
#                 for cmd in self.commands])