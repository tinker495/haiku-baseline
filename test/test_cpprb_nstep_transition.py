import numpy as np

from jax_baselines.common.cpprb_buffers import (
    MultiPrioritizedReplayBuffer,
    ReplayBuffer,
)

n = 4
gamma = 0.995
mprb = MultiPrioritizedReplayBuffer(200, [[4]], 0.6, 1, n_step=n, gamma=gamma)
gloabal_buffer, env_dict, n_s = mprb.buffer_info()
rb = ReplayBuffer(100, env_dict=env_dict, n_s=n_s)
true_done = False

for i in range(11):
    if i % 20 == 10:
        rb.add([np.arange(i, i + 4)], 1, 1, [np.arange(i + 4, i + 8)], true_done, False)
        rb.episode_end()
    else:
        rb.add([np.arange(i, i + 4)], 1, 1, [np.arange(i + 4, i + 8)], False, False)
#

transition = rb.get_buffer()
rb.clear()

for i in range(len(transition["terminated"])):
    st = ""
    for k in transition.keys():
        st += f"{k}: {transition[k][i]}, "
    print(st)

for i in range(n, 0, -1):
    d = transition[k][-i]
    print(d, (1 - d) * np.power(gamma, n), " = ", np.power(gamma, i))

print((1 - true_done) * np.power(gamma, n))
