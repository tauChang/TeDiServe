# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from concurrent.futures import Future
import inspect
from typing import Optional, Union

from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.executor.ray_distributed_executor import (  # noqa
    RayDistributedExecutor as RayDistributedExecutorV0)
from vllm.logger import init_logger
from vllm.v1.engine import ReconfigureDistributedRequest, ReconfigureRankType
from vllm.v1.executor.abstract import Executor
from vllm.v1.outputs import ModelRunnerOutput

logger = init_logger(__name__)


class FutureWrapper(Future):
    """A wrapper around Ray output reference to meet the interface
    of .execute_model(): The top level (core busy loop) expects .result() api 
    to block and return a single output.
    
    If aggregator is provided, the outputs from all workers are aggregated upon 
    the result() call. If not only the first worker's output is returned.
    """

    def __init__(self, refs, aggregator: Optional[KVOutputAggregator] = None):
        super().__init__()
        self.refs = refs
        self.aggregator = aggregator

    def result(self, timeout=None):
        if timeout is not None:
            raise NotImplementedError("timeout is not supported")

        if self.aggregator is None:
            return self.refs[0].get()

        outputs = [ref.get() for ref in self.refs]
        return self.aggregator.aggregate(outputs, output_rank=0)


class AsyncFutureWrapper(Future):
    """A wrapper around Ray output reference to meet the interface
    of .execute_model(): The top level (core busy loop) expects .result() api 
    to block and return a single output.
    
    If aggregator is provided, the outputs from all workers are aggregated upon 
    the result() call. If not only the first worker's output is returned.
    """

    def __init__(self, refs, aggregator: Optional[KVOutputAggregator] = None):
        super().__init__()
        self.refs = refs
        self.aggregator = aggregator

    async def result(self):
        if self.aggregator is None:
            return await self.refs[0]

        outputs = await asyncio.gather(*self.refs)
        return self.aggregator.aggregate(outputs, output_rank=0)
class RayDistributedExecutor(RayDistributedExecutorV0, Executor):
    """Ray distributed executor using Ray Compiled Graphs."""

    def _init_executor(self) -> None:
        super()._init_executor()

        # KV connector setup
        self.has_connector = self.vllm_config.kv_transfer_config is not None
        self.kv_output_aggregator = KVOutputAggregator(
            self.parallel_config.world_size)

    @property
    def max_concurrent_batches(self) -> int:
        """Ray distributed executor supports pipeline parallelism,
        meaning that it allows PP size batches to be executed concurrently.
        """
        if self.scheduler_config.async_scheduling:
            return 2
        return self.parallel_config.pipeline_parallel_size

    def execute_model(
        self,
        scheduler_output,
    ) -> Union[ModelRunnerOutput, Future[ModelRunnerOutput]]:
        """Execute the model on the Ray workers.

        Args:
            scheduler_output: The scheduler output to execute.

        Returns:
            The model runner output.
        """
        # Build the compiled DAG for the first time.
        if self.forward_dag is None:  # type: ignore
            self.forward_dag = self._compiled_ray_dag(enable_asyncio=False)

        refs = self.forward_dag.execute(scheduler_output)  # type: ignore

        if not self.has_connector:
            # Get output only from a single worker (output_rank)
            # When PP is not used, we block here until the result is available.
            if self.max_concurrent_batches == 1:
                return refs[0].get()

            # When PP is used, we return a FutureWrapper immediately so that
            # the scheduler can yield to the next batch.
            return FutureWrapper(refs)

        # Get output from all workers when connector is present
        if self.max_concurrent_batches == 1:
            # Block and get results from all workers
            outputs = [ref.get() for ref in refs]
            return self.kv_output_aggregator.aggregate(outputs)

        # Return a future that will aggregate outputs from all workers
        return FutureWrapper(refs, self.kv_output_aggregator)

    async def execute_model_async(
        self,
        scheduler_output,
    ) -> Union[ModelRunnerOutput, Future[ModelRunnerOutput]]:
        """Execute the model on the Ray workers.

        Args:
            scheduler_output: The scheduler output to execute.

        Returns:
            The model runner output.
        """
        # Build the compiled DAG for the first time.
        if self.forward_dag is None:  # type: ignore
            self.forward_dag = self._compiled_ray_dag(enable_asyncio=True)

        refs = await self.forward_dag.execute_async(scheduler_output)

        # Some Ray compiled-DAG async paths return a singleton container with
        # an awaitable payload. Normalize to a list of per-worker awaitables.
        if isinstance(refs, list) and len(refs) == 1 and inspect.isawaitable(
                refs[0]):
            refs = await refs[0]

        if not isinstance(refs, list):
            refs = [refs]

        if not self.has_connector:
            # Get output only from a single worker (output_rank)
            # When PP is not used, we block here until the result is available.
            if self.max_concurrent_batches == 1:
                first = refs[0]
                return await first if inspect.isawaitable(first) else first

            # When PP is used, we return a FutureWrapper immediately so that
            # the scheduler can yield to the next batch.
            return AsyncFutureWrapper(refs)

        # Get output from all workers when connector is present
        if self.max_concurrent_batches == 1:
            # Block and get results from all workers
            outputs = [await ref if inspect.isawaitable(ref) else ref
                       for ref in refs]
            return self.kv_output_aggregator.aggregate(outputs)

        # Return a future that will aggregate outputs from all workers
        return AsyncFutureWrapper(refs, self.kv_output_aggregator)


    def reinitialize_distributed(
            self, reconfig_request: ReconfigureDistributedRequest) -> None:
        self._run_workers("reinitialize_distributed", reconfig_request)
        if reconfig_request.new_data_parallel_rank == \
        ReconfigureRankType.SHUTDOWN_CURRENT_RANK:
            self.shutdown()
        return