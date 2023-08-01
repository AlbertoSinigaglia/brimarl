from brimarl_masked.main.train_mappo_parallel import main as mappo_parallel_main
from brimarl_masked.main.train_a2c import main as a2c_main
from brimarl_masked.main.train_ppo import main as ppo_main
from brimarl_masked.main.train_mappo import main as mappo_main

if __name__ == "__main__":
    import time
    t = time.time()
    ppo_main()
    print("\n\n\n\n\n")
    print(time.time()-t)