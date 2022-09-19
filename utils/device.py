__all__ = ['gpu_to_cpu', 'cpu_to_gpu']


def gpu_to_cpu(current_state):
    for k, v in current_state.items():
        current_state[k] = v.cpu()
    return current_state


def cpu_to_gpu(current_state, device):
    for k, v in current_state.items():
        current_state[k] = v.to(device)
    return current_state