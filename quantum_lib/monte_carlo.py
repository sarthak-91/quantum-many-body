import torch

def monte_carlo_sample(input_vector, probab, update_rule, domain_check=None, n_samples=1000, burn_in=1000, sample_gap=50):
    """
    Monte Carlo sampler .
    """
    current_state = input_vector.clone().detach().requires_grad_(input_vector.requires_grad)
    samples = []
    
    prob_current = probab(current_state)

    for step in range(n_samples + burn_in):
        proposed_state = update_rule(current_state)

        if domain_check and not domain_check(proposed_state):
            continue

        prob_proposed = probab(proposed_state)
        acceptance_ratio = prob_proposed / prob_current

        if torch.rand(1, device=input_vector.device) < acceptance_ratio:
            current_state = proposed_state
            prob_current = prob_proposed

        if step >= burn_in and step % sample_gap == 0:
            samples.append(current_state) 

    return torch.stack(samples)
