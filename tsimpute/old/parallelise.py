import concurrent.futures
import heapq


# Code from Ben Jeffery to parallelise independent runs of a function.

def threaded_map(func, args, num_workers):
    results_buffer = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = set()
        next_index = 0
        for i, arg in enumerate(args):
            # +1 so that we're not waiting for the args generator to produce the next arg
            while len(futures) >= num_workers + 1:
                # If there are too many in-progress tasks, wait for one to complete
                done, futures = concurrent.futures.wait(
                    futures, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    index, result = future.result()
                    if index == next_index:
                        # If this result is the next expected one, yield it immediately
                        yield result
                        next_index += 1
                    else:
                        heapq.heappush(results_buffer, (index, result))
                    # Yield any results from the buffer that are next in line
                    while results_buffer and results_buffer[0][0] == next_index:
                        _, result = heapq.heappop(results_buffer)
                        yield result
                        next_index += 1
            # Wraps the function so we can track the index of the argument
            futures.add(executor.submit(lambda arg, i=i: (i, func(arg)), arg))
        concurrent.futures.wait(futures)
        for future in futures:
            index, result = future.result()
            if index == next_index:
                yield result
                next_index += 1
            else:
                heapq.heappush(results_buffer, (index, result))
        # Yield any remaining results in the buffer
        while results_buffer:
            _, result = heapq.heappop(results_buffer)
            yield result
