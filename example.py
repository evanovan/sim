import simulus

def run_simulation():
    env = simulus.simulator()

    # Parameters
    data_bandwidth = 5e9       # 5 GB/s
    data_size = 100e6         # 100 MB
    mcu_compute_time = 0.05   # seconds

    data_done = [False]  # Shared flag instead of event

    def data_transfer():
        transfer_time = data_size / data_bandwidth
        print(f"[{env.now:.6f}] data: Start transfer ({data_size/1e6:.1f} MB @ {data_bandwidth/1e9:.1f} GB/s)")
        env.sleep(transfer_time)
        data_done[0] = True
        print(f"[{env.now:.6f}] data: Transfer complete (duration: {transfer_time:.6f} s)")

    def m_compute():
        while not data_done[0]:
            env.sleep(0.001)  # Polling interval (1 ms)
        print(f"[{env.now:.6f}] MCU: Start m multiplication")
        env.sleep(mcu_compute_time)
        print(f"[{env.now:.6f}] MCU: Finished computation (duration: {mcu_compute_time:.6f} s)")

    env.process(data_transfer)
    env.process(m_compute)

    env.run()

    print(f"\n✅ Total simulated latency = {env.now:.6f} seconds")

if __name__ == "__main__":
    run_simulation()
