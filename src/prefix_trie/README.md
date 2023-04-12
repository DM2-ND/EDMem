# Prefix Trees

This directory includes scripts to build prefix trees for static/dynamic entity linking.

For static entity linking, run `build_entity_prefix_trie.py` to build a prefix tree based on the predicted entities from an entity linking model. The prefix tree tells the model which tokens are allowed to generate in the constrained generation. For example:

```bash
python src/prefix_trie/build_entity_trie_each_instance.py \
    -el_result ${CKPT_DIR}/predictions.json \
    -entid2entityitemid data/wikipedia/entid2entityitemid_1M.json \
    -topk 1 \
    -lowercase \
    -output ${CKPT_DIR}/top1_entity_trie.pkl
```

To select multiple candidate entities for downstream generation, increase `topk`. To also include entity aliases in the prefix tree, run `build_mention_trie_each_instance.py`.

For dynamic entity linking in generation tasks, theoretically we don't really need a prefix tree. But because the prefix_trie argument is required for activating the contrained generation, we pass a dummy tree object to the code (works like a placeholder). To build the dummy tree, run `build_dummy_trie.py`.