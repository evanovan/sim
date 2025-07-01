import time
import random
import simpy
import simulus
import statistics
import multiprocessing
from functools import partial

# Define simulation parameters
NUM_TASKS = 100000*2  # Total number of tasks to simulate
NUM_SMs = 40  # Number of Processing Elements (workers)
NUM_TRIALS = 1  # Number of simulation trials to average results
NUM_SIMULATORS = min(multiprocessing.cpu_count(), 4)  # Number of Simulus simulators, capSMd at 4 or CPU count
NUM_TASKS_SMR_SIM = NUM_TASKS // NUM_SIMULATORS  # Tasks SMr Simulus simulator
NUM_SMs_SMR_SIM = max(1, NUM_SMs // NUM_SIMULATORS)  # SMs SMr Simulus simulator

def generate_tasks(seed):
    """Generates a list of tasks with random durations.

    Args:
        seed (int): Seed for the random number generator to ensure reproducibility.

    Returns:
        list: A list of tuples, where each tuple contains (task_id, duration).
    """
    random.seed(seed)
    return [(i, random.uniform(1, 10)) for i in range(NUM_TASKS)]

def run_simpy(seed, tasks):
    """Runs the simulation using SimPy.

    Args:
        seed (int): Seed for the random number generator.
        tasks (list): A list of tasks to be processed.

    Returns:
        tuple: (wall_time, sim_time, queue_length)
            wall_time (float): Wall clock time taken for the simulation.
            sim_time (float): Simulation time.
            queue_length (int): Number of tasks remaining in the queue at the end of the simulation.
    """
    print("\n--- SimPy Simulation ---")
    start_wall = time.time()  # Record start time
    env = simpy.Environment()  # Create SimPy environment
    task_store = simpy.Store(env, capacity=NUM_TASKS)  # Task queue
    gen_done = [False]  # Flag to indicate if task generation is complete
    event_count = [0]  # Counter for events

    def task_generator(env):
        """Generates tasks and adds them to the task queue."""
        for i, (task_id, duration) in enumerate(tasks):
            if duration <= 0:
                print(f"[{env.now:.2f}] Warning: Invalid task duration for Task {task_id}")
                continue
            yield task_store.put((task_id, duration))  # Put task in queue
            event_count[0] += 1
            if i < 10:
                print(f"[{env.now:.2f}] Task {task_id} generated (duration {duration:.2f})")
            yield env.timeout(0.01)  # Simulate task generation delay
            event_count[0] += 1
        gen_done[0] = True  # Mark task generation as complete
        print(f"[{env.now:.2f}] Task generation complete")

    def processing_element(env, SM_id):
        """Simulates a processing element consuming tasks from the queue."""
        while not gen_done[0] or task_store.items:  # Continue until all tasks are generated and queue is empty
            try:
                task = yield task_store.get()  # Get task from queue
                event_count[0] += 1
                task_id, duration = task
                if task_id < 10:
                    print(f"[{env.now:.2f}] SM{SM_id} starts Task {task_id} (duration {duration:.2f})")
                yield env.timeout(duration)  # Simulate processing time
                event_count[0] += 1
                if task_id < 10:
                    print(f"[{env.now:.2f}] SM{SM_id} finishes Task {task_id}")
            except simpy.Interrupt:
                if gen_done[0] and not task_store.items:
                    break

    env.process(task_generator(env))  # Start task generator process
    for i in range(NUM_SMs):
        env.process(processing_element(env, i))  # Start processing element processes

    env.run()  # Run the simulation
    sim_time = env.now  # Get simulation time
    wall_time = time.time() - start_wall  # Calculate wall time
    print(f"\n✅ SimPy: Simulation time = {sim_time:.2f}, Wall time = {wall_time:.4f} sec, Events = {event_count[0]}")
    return wall_time, sim_time, len(task_store.items)

def run_single_simulator(args):
    """Runs a single Simulus simulator.

    Args:
        args (tuple): A tuple containing (sim_idx, seed, tasks, start, end, num_SMs).
            sim_idx (int): Index of the simulator.
            seed (int): Seed for the random number generator.
            tasks (list): List of tasks.
            start (int): Start index of tasks for this simulator.
            end (int): End index of tasks for this simulator.
            num_SMs (int): Number of processing elements for this simulator.

    Returns:
        tuple: (sim_time, event_count, queue_length)
            sim_time (float): Simulation time.
            event_count (int): Number of events processed.
            queue_length (int): Number of tasks remaining in the queue.
    """
    sim_idx, seed, tasks, start, end, num_SMs = args
    sim = simulus.simulator(f'sim{sim_idx}')  # Create a Simulus simulator
    task_queue = []  # Task queue for this simulator
    gen_done = [False]  # Flag for task generation completion
    task_sem = sim.semaphore(0)  # Semaphore for task synchronization
    event_count = [0]  # Event counter

    def task_generator():
        """Generates tasks and adds them to the task queue."""
        for j in range(start, min(end, len(tasks))):
            task_id, duration = tasks[j]
            if duration <= 0:
                print(f"[{sim.now:.2f}] Warning: Invalid task duration for Task {task_id}")
                continue
            task_queue.append((task_id, duration))  # Add task to queue
            task_sem.signal()  # Signal that a task is available
            event_count[0] += 1
            if task_id < 10:
                print(f"[{sim.now:.2f}] Task {task_id} generated (duration {duration:.2f})")
            sim.sleep(0.01)  # Simulate task generation delay
            event_count[0] += 1
        gen_done[0] = True  # Mark task generation as complete
        task_sem.signal()  # Signal to wake up any waiting SMs

    def processing_element(SM_id):
        """Simulates a processing element consuming tasks from the queue."""
        while not gen_done[0] or task_queue:  # Continue until all tasks are generated and queue is empty
            task_sem.wait()  # Wait for a task to become available
            event_count[0] += 1
            if task_queue:
                task_id, duration = task_queue.pop(0)  # Get task from queue
                if task_id < 10:
                    print(f"[{sim.now:.2f}] SM{SM_id} starts Task {task_id} (duration {duration:.2f})")
                sim.sleep(duration)  # Simulate processing time
                event_count[0] += 1
                if task_id < 10:
                    print(f"[{sim.now:.2f}] SM{SM_id} finishes Task {task_id}")
            elif gen_done[0]:
                break

    sim.process(task_generator)  # Start task generator process
    for k in range(num_SMs):
        sim.process(lambda k=k: processing_element(k))  # Start processing element processes

    sim.run()  # Run the simulation
    return sim.now, event_count[0], len(task_queue)  # Return simulation results

def run_simulus(seed, tasks):
    """Runs the simulation using Simulus with multiprocessing.

    Args:
        seed (int): Seed for the random number generator.
        tasks (list): A list of tasks to be processed.

    Returns:
        tuple: (wall_time, sim_time, queue_length)
            wall_time (float): Wall clock time taken for the simulation.
            sim_time (float): Simulation time.
            queue_length (int): Number of tasks remaining in the queues.
    """
    print("\n--- Simulus Simulation ---")
    random.seed(seed)
    start_wall = time.time()  # Record start time

    with multiprocessing.Pool(processes=NUM_SIMULATORS) as pool:  # Create a pool of processes
        args = [(i, seed, tasks, i*NUM_TASKS_SMR_SIM, (i+1)*NUM_TASKS_SMR_SIM, NUM_SMs_SMR_SIM) for i in range(NUM_SIMULATORS)]
        results = pool.map(run_single_simulator, args)  # Run simulations in parallel

    sim_time = max(result[0] for result in results)  # Get maximum simulation time
    event_count = sum(result[1] for result in results)  # Sum the event counts
    queue_length = sum(result[2] for result in results)  # Sum the queue lengths
    wall_time = time.time() - start_wall  # Calculate wall time

    print(f"\n✅ Simulus: Simulation time = {sim_time:.2f}, Wall time = {wall_time:.4f} sec, Events = {event_count}")
    return wall_time, sim_time, queue_length

if __name__ == "__main__":
    print(f"Simulus version: {simulus.__version__}")
    SEED = 42  # Random seed for reproducibility
    tasks = generate_tasks(SEED)  # Generate tasks
    simpy_wall_times = []
    simulus_wall_times = []
    simpy_sim_time = None
    simulus_sim_time = None

    for trial in range(NUM_TRIALS):
        print(f"\n=== Trial {trial + 1} ===")
        w1, s1, q1 = run_simpy(SEED, tasks)  # Run SimPy simulation
        simpy_wall_times.append(w1)
        simpy_sim_time = s1
        print(f"Tasks remaining: {q1}")
        
        w2, s2, q2 = run_simulus(SEED, tasks)  # Run Simulus simulation
        simulus_wall_times.append(w2)
        simulus_sim_time = s2
        print(f"Tasks remaining: {q2}")

    avg_simpy_wall = statistics.mean(simpy_wall_times)  # Calculate average wall time for SimPy
    avg_simulus_wall = statistics.mean(simulus_wall_times)  # Calculate average wall time for Simulus
    std_simpy_wall = statistics.stdev(simpy_wall_times) if len(simpy_wall_times) > 1 else 0  # Calculate standard deviation for SimPy
    std_simulus_wall = statistics.stdev(simulus_wall_times) if len(simulus_wall_times) > 1 else 0  # Calculate standard deviation for Simulus
    print(f"\nSummary:")
    print(f"SimPy: Average Wall time = {avg_simpy_wall:.4f}s (±{std_simpy_wall:.4f}s), Simulation time = {simpy_sim_time:.2f}s")
    print(f"Simulus: Average Wall time = {avg_simulus_wall:.4f}s (±{std_simulus_wall:.4f}s), Simulation time = {simulus_sim_time:.2f}s")
    print(f"SSMedup (SimPy/Simulus): {avg_simpy_wall / avg_simulus_wall:.2f}x")  # Calculate sSMedup
