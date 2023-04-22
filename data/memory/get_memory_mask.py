import argparse
import torch
import json

ENTITY_VOCAB_SIZE = 1000010


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mask_rate', default=0.8, type=float, help='Mask ratio of the entity memory')
    parser.add_argument('-output', default='memory_mask.json', help='Output path')
    args = parser.parse_args()
    assert 0.0 < args.mask_rate <= 1.0

    # 0: unmasked, 1: masked
    if args.mask_rate == 1.0:
        entity_mask = torch.ones(ENTITY_VOCAB_SIZE).long()
    else:
        entity_mask = torch.zeros(ENTITY_VOCAB_SIZE).long()
        probability_matrix = torch.full((ENTITY_VOCAB_SIZE,), args.mask_rate)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        entity_mask[masked_indices] = 1

    num_masked = torch.sum(entity_mask).item()
    print(f'Number of masked entities: {num_masked}')
    torch.save(entity_mask, args.output)


if __name__ == "__main__":
    main()
