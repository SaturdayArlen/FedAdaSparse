import torch


def get_client_params(model, queue, activated):
    params = {}
    for idx in activated:
        lora = queue[idx]
        params[idx] = {
            'encoder': lora.encoder.weight.data.clone(),
            'decoder': lora.decoder.weight.data.clone()
        }
    return params


def set_client_params(model, queue, activated, global_params):
    for idx in activated:
        if idx in global_params:
            queue[idx].encoder.weight.data = global_params[idx]['encoder'].clone()
            queue[idx].decoder.weight.data = global_params[idx]['decoder'].clone()


def get_lora_param_count(lora_module):
    return lora_module.encoder.weight.numel() + lora_module.decoder.weight.numel()


def get_client_communication_size(queue, activated):
    total = 0
    for idx in activated:
        total += get_lora_param_count(queue[idx])
    return total * 4


def aggregate_layer_wise(client_updates):
    layer_sums = {}
    layer_counts = {}
    for client in client_updates:
        for layer_idx in client['activated']:
            if layer_idx not in client['params']:
                continue
            if layer_idx not in layer_sums:
                layer_sums[layer_idx] = {
                    'encoder': client['params'][layer_idx]['encoder'].clone(),
                    'decoder': client['params'][layer_idx]['decoder'].clone()
                }
                layer_counts[layer_idx] = 1
            else:
                layer_sums[layer_idx]['encoder'] += client['params'][layer_idx]['encoder']
                layer_sums[layer_idx]['decoder'] += client['params'][layer_idx]['decoder']
                layer_counts[layer_idx] += 1
    global_params = {}
    for layer_idx in layer_sums:
        global_params[layer_idx] = {
            'encoder': layer_sums[layer_idx]['encoder'] / layer_counts[layer_idx],
            'decoder': layer_sums[layer_idx]['decoder'] / layer_counts[layer_idx]
        }
    return global_params


def aggregate_layer_wise_weighted(client_updates):
    layer_sums = {}
    layer_total_weight = {}

    for client in client_updates:
        weight = client.get('num_samples', 1)
        for layer_idx in client['activated']:
            if layer_idx not in client['params']:
                continue
            if layer_idx not in layer_sums:
                layer_sums[layer_idx] = {
                    'encoder': client['params'][layer_idx]['encoder'].clone() * weight,
                    'decoder': client['params'][layer_idx]['decoder'].clone() * weight
                }
                layer_total_weight[layer_idx] = weight
            else:
                layer_sums[layer_idx]['encoder'] += client['params'][layer_idx]['encoder'] * weight
                layer_sums[layer_idx]['decoder'] += client['params'][layer_idx]['decoder'] * weight
                layer_total_weight[layer_idx] += weight

    global_params = {}
    for layer_idx in layer_sums:
        global_params[layer_idx] = {
            'encoder': layer_sums[layer_idx]['encoder'] / layer_total_weight[layer_idx],
            'decoder': layer_sums[layer_idx]['decoder'] / layer_total_weight[layer_idx]
        }
    return global_params
